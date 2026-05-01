from typing import Any, Dict, List

from .prompts import (
    PLANNER_PROMPT,
    DATA_AGENT_PROMPT,
    INSIGHT_AGENT_PROMPT,
    INSIGHT_RETRY_PROMPT,
    EVALUATOR_AGENT_PROMPT,
    EVALUATOR_RETRY_PROMPT,
    CREATIVE_AGENT_PROMPT,
)
from .utils import call_llm, safe_json_loads, build_data_summary


def run_planner(user_question: str) -> Dict[str, Any]:
    user_prompt = f"User question:\n\"\"\"{user_question}\"\"\"\n\nProduce the plan JSON."
    raw = call_llm(PLANNER_PROMPT, user_prompt)
    return safe_json_loads(raw)


def run_data_agent(summary_stats: Dict[str, Any],
                   task_description: str = "") -> Dict[str, Any]:
    """
    Here the 'system code' has already computed summary_stats (build_data_summary).
    We send that to the LLM so it can express a human-readable summary.
    task_description: focused directive from the Planner Agent.
    """
    directive = (
        f"PLANNER DIRECTIVE:\n{task_description}\n\n"
        if task_description else ""
    )
    user_prompt = (
        f"{directive}"
        "You are given compact performance statistics as JSON.\n"
        "Use them to summarize what is happening in the account.\n\n"
        f"DATA SUMMARY (JSON):\n```json\n{summary_stats}\n```"
    )
    raw = call_llm(DATA_AGENT_PROMPT, user_prompt)
    return safe_json_loads(raw)


def run_insight_agent(user_question: str,
                      data_summary: Dict[str, Any],
                      data_agent_view: Dict[str, Any],
                      task_description: str = "") -> Dict[str, Any]:
    """
    task_description: focused directive from the Planner Agent specifying
    which dimensions (audience, creative, geo, platform) to prioritise.
    """
    directive = (
        f"PLANNER DIRECTIVE (prioritise these areas):\n{task_description}\n\n"
        if task_description else ""
    )
    user_prompt = (
        f"{directive}"
        f"User question:\n\"\"\"{user_question}\"\"\"\n\n"
        "System numeric summary:\n"
        f"```json\n{data_summary}\n```\n\n"
        "Data Agent textual summary:\n"
        f"```json\n{data_agent_view}\n```"
    )
    raw = call_llm(INSIGHT_AGENT_PROMPT, user_prompt)
    return safe_json_loads(raw)


def run_insight_agent_with_retry(
        user_question: str,
        data_summary: Dict[str, Any],
        data_agent_view: Dict[str, Any],
        task_description: str = "",
        max_retries: int = 2) -> Dict[str, Any]:
    """
    Runs the Insight Agent and automatically triggers a reflection/refinement
    pass for any hypothesis returned with confidence == "low".
    Retries up to max_retries times; returns the best result achieved.
    """
    result = run_insight_agent(user_question, data_summary, data_agent_view,
                               task_description=task_description)

    for attempt in range(1, max_retries + 1):
        low_conf = [
            h for h in result.get("insights", [])
            if h.get("confidence", "").lower() == "low"
        ]
        if not low_conf:
            break  # all hypotheses are medium/high — no retry needed

        low_ids = [h.get("id", "?") for h in low_conf]
        user_prompt = (
            f"RETRY ATTEMPT {attempt} of {max_retries}.\n"
            f"Low-confidence hypothesis IDs that need refinement: {low_ids}\n\n"
            "PREVIOUS FULL OUTPUT:\n"
            f"```json\n{result}\n```\n\n"
            "ORIGINAL DATA SUMMARIES:\n"
            f"System numeric summary:\n```json\n{data_summary}\n```\n\n"
            f"Data Agent textual summary:\n```json\n{data_agent_view}\n```\n\n"
            "Refine ALL low-confidence hypotheses and return the complete improved output."
        )
        raw = call_llm(INSIGHT_RETRY_PROMPT, user_prompt)
        refined = safe_json_loads(raw)

        # Accept the refinement only if it produced valid insights
        if refined.get("insights"):
            result = refined

    return result


def run_evaluator_agent(insights: Dict[str, Any],
                        evaluation_stats: Dict[str, Any],
                        task_description: str = "") -> Dict[str, Any]:
    """
    evaluation_stats can be derived from data_summary (e.g. deltas, top movers).
    task_description: focused directive from the Planner specifying which
    hypotheses or segments to scrutinise most carefully.
    """
    directive = (
        f"PLANNER DIRECTIVE (focus your validation on):\n{task_description}\n\n"
        if task_description else ""
    )
    user_prompt = (
        f"{directive}"
        "You are given hypotheses and quantitative stats.\n\n"
        f"HYPOTHESES JSON:\n```json\n{insights}\n```\n\n"
        f"QUANTITATIVE STATS JSON:\n```json\n{evaluation_stats}\n```"
    )
    raw = call_llm(EVALUATOR_AGENT_PROMPT, user_prompt)
    return safe_json_loads(raw)


def run_evaluator_agent_with_retry(
        insights: Dict[str, Any],
        evaluation_stats: Dict[str, Any],
        task_description: str = "",
        max_retries: int = 2) -> Dict[str, Any]:
    """
    Runs the Evaluator Agent and triggers a reflection/refinement pass for
    any evaluation returned with confidence == "low".
    Retries up to max_retries times; returns the best result achieved.
    """
    result = run_evaluator_agent(insights, evaluation_stats,
                                 task_description=task_description)

    for attempt in range(1, max_retries + 1):
        low_conf = [
            e for e in result.get("evaluations", [])
            if e.get("confidence", "").lower() == "low"
        ]
        if not low_conf:
            break  # all evaluations are medium/high — no retry needed

        low_ids = [e.get("hypothesis_id", "?") for e in low_conf]
        user_prompt = (
            f"RETRY ATTEMPT {attempt} of {max_retries}.\n"
            f"Low-confidence evaluation IDs that need refinement: {low_ids}\n\n"
            "PREVIOUS FULL OUTPUT:\n"
            f"```json\n{result}\n```\n\n"
            "ORIGINAL HYPOTHESES:\n"
            f"```json\n{insights}\n```\n\n"
            "QUANTITATIVE STATS:\n"
            f"```json\n{evaluation_stats}\n```\n\n"
            "Re-examine the stats carefully and return the complete improved evaluation output."
        )
        raw = call_llm(EVALUATOR_RETRY_PROMPT, user_prompt)
        refined = safe_json_loads(raw)

        # Accept the refinement only if it produced valid evaluations
        if refined.get("evaluations"):
            result = refined

    return result


def run_creative_agent(creative_context: Dict[str, Any],
                       task_description: str = "") -> Dict[str, Any]:
    """
    creative_context should contain low_ctr_campaigns and high_ctr_creatives.
    task_description: focused directive from the Planner specifying tone,
    audience angle, or creative strategy to pursue.
    """
    directive = (
        f"PLANNER DIRECTIVE (creative strategy focus):\n{task_description}\n\n"
        if task_description else ""
    )
    user_prompt = (
        f"{directive}"
        "You are given context for low-CTR and high-CTR creatives as JSON.\n"
        "Propose new creatives following the output schema.\n\n"
        f"CREATIVE CONTEXT JSON:\n```json\n{creative_context}\n```"
    )
    raw = call_llm(CREATIVE_AGENT_PROMPT, user_prompt)
    return safe_json_loads(raw)


def build_evaluation_stats(data_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes additional statistics to help the Evaluator Agent confidently
    validate or reject hypotheses. Adds explicit rankings and trend data.
    """
    eval_stats: Dict[str, Any] = {"base_summary": data_summary}

    # 1. Overall ROAS Trend
    roas_by_date = data_summary.get("roas_by_date", [])
    if len(roas_by_date) >= 2:
        first = roas_by_date[0]
        last = roas_by_date[-1]
        start_roas = first.get("roas", 0.0)
        end_roas = last.get("roas", 0.0)
        delta = end_roas - start_roas
        pct_change = (delta / start_roas * 100.0) if start_roas != 0 else 0.0
        
        # Calculate volatility (min and max ROAS during the period)
        all_roas = [day.get("roas", 0.0) for day in roas_by_date]
        
        eval_stats["roas_trend"] = {
            "start_date": first.get("date"),
            "start_roas": start_roas,
            "end_date": last.get("date"),
            "end_roas": end_roas,
            "absolute_change": delta,
            "pct_change": pct_change,
            "period_max_roas": max(all_roas) if all_roas else 0.0,
            "period_min_roas": min(all_roas) if all_roas else 0.0,
        }

    # 2. Campaign Rankings (ROAS & CTR)
    # Using top_campaigns_by_spend as a proxy for the most impactful campaigns
    campaigns = data_summary.get("top_campaigns_by_spend", [])
    if campaigns:
        # Sort by ROAS
        sorted_by_roas = sorted(campaigns, key=lambda x: x.get("roas", 0.0), reverse=True)
        eval_stats["top_3_campaigns_by_roas"] = sorted_by_roas[:3]
        eval_stats["bottom_3_campaigns_by_roas"] = sorted_by_roas[-3:][::-1] # Lowest first

        # Sort by CTR
        sorted_by_ctr = sorted(campaigns, key=lambda x: x.get("ctr", 0.0), reverse=True)
        eval_stats["top_3_campaigns_by_ctr"] = sorted_by_ctr[:3]
        eval_stats["bottom_3_campaigns_by_ctr"] = sorted_by_ctr[-3:][::-1] # Lowest first

    eval_stats["note"] = (
        "Use roas_trend (including pct_change and volatility) and the top/bottom "
        "campaign rankings to rigorously validate hypotheses. Cite these specific "
        "numbers when justifying your confidence scores."
    )
    return eval_stats


def build_creative_context(data_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "low_ctr_campaigns": data_summary.get("low_ctr_campaigns", []),
        "high_ctr_creatives": data_summary.get("high_ctr_creatives", []),
    }
