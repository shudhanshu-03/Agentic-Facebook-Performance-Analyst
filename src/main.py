import os
from typing import Dict, Any

from .utils import (
    load_facebook_data,
    build_data_summary,
    save_json,
    append_log,
)
from .agents import (
    run_planner,
    run_data_agent,
    run_insight_agent_with_retry,
    run_evaluator_agent_with_retry,
    run_creative_agent,
    build_evaluation_stats,
    build_creative_context,
)


DATA_PATH = os.getenv("FB_DATA_PATH", "data/synthetic_fb_ads_undergarments.csv")
OUTPUT_DIR = "outputs"
INSIGHTS_PATH = os.path.join(OUTPUT_DIR, "insights.json")
CREATIVES_PATH = os.path.join(OUTPUT_DIR, "creatives.json")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs.txt")

def _get_task(plan: Dict[str, Any], target_agent: str) -> str:
    """
    Extract the Planner's task description for a given agent name.
    Falls back to an empty string so agents work even if the plan is malformed.
    """
    for task in plan.get("subtasks", []):
        if task.get("target_agent", "").lower() == target_agent.lower():
            return task.get("description", "")
    return ""

def run_pipeline(user_question: str) -> Dict[str, Any]:
    append_log(LOG_PATH, f"=== New run ===")
    append_log(LOG_PATH, f"Question: {user_question}")

    # 1. Load data
    df = load_facebook_data(DATA_PATH)
    append_log(LOG_PATH, f"Loaded dataframe with {len(df)} rows.")

    # 2. Planner Agent — decomposes the question into focused subtasks
    plan = run_planner(user_question)
    append_log(LOG_PATH, f"Planner output: {plan}")
    append_log(LOG_PATH, f"Clarified question: {plan.get('clarified_question', user_question)}")
    for task in plan.get("subtasks", []):
        append_log(LOG_PATH, f"  [{task.get('id')}] → {task.get('target_agent')}: {task.get('description')}")

    # 3. Build numeric summary (system code)
    data_summary = build_data_summary(df)
    append_log(LOG_PATH, "Built numeric data summary.")

    # 4. Data Agent — Planner directs which metrics to emphasise
    data_task = _get_task(plan, "data")
    data_agent_view = run_data_agent(data_summary, task_description=data_task)
    append_log(LOG_PATH, f"Data Agent view: {data_agent_view}")

    # 5. Insight Agent — with automatic reflection/retry for low-confidence hypotheses
    insight_task = _get_task(plan, "insight")
    insights = run_insight_agent_with_retry(
        user_question, data_summary, data_agent_view,
        task_description=insight_task, max_retries=2
    )
    low_h = [h for h in insights.get("insights", []) if h.get("confidence") == "low"]
    append_log(LOG_PATH, f"Insight Agent output: {insights}")
    append_log(LOG_PATH,
               f"Insight Agent: {len(insights.get('insights', []))} hypotheses, "
               f"{len(low_h)} still low-confidence after retries.")

    # 6. Evaluator Agent — with automatic reflection/retry for low-confidence evaluations
    eval_stats = build_evaluation_stats(data_summary)
    evaluator_task = _get_task(plan, "evaluator")
    evaluations = run_evaluator_agent_with_retry(
        insights, eval_stats,
        task_description=evaluator_task, max_retries=2
    )
    low_e = [e for e in evaluations.get("evaluations", []) if e.get("confidence") == "low"]
    append_log(LOG_PATH, f"Evaluator Agent output: {evaluations}")
    append_log(LOG_PATH,
               f"Evaluator Agent: {len(evaluations.get('evaluations', []))} evaluations, "
               f"{len(low_e)} still low-confidence after retries.")

    # 7. Creative Agent — Planner directs creative strategy angle
    creative_context = build_creative_context(data_summary)
    creative_task = _get_task(plan, "creative")
    creatives = run_creative_agent(creative_context, task_description=creative_task)
    append_log(LOG_PATH, f"Creative Agent output: {creatives}")

    # 8. Save outputs
    insights_output = {
        "question": user_question,
        "plan": plan,
        "data_summary": data_summary,
        "data_agent_view": data_agent_view,
        "insights": insights,
        "evaluations": evaluations,
    }
    save_json(INSIGHTS_PATH, insights_output)
    save_json(CREATIVES_PATH, creatives)

    append_log(LOG_PATH, "Saved insights.json and creatives.json")
    return {
        "insights": insights_output,
        "creatives": creatives,
    }


if __name__ == "__main__":
    # You can change this default question or accept input from CLI.
    default_question = (
        "Diagnose why ROAS has changed over the last 30 days and "
        "recommend new creative ideas for low-CTR campaigns."
    )
    result = run_pipeline(default_question)
    print("Run complete.")
    print(f"Insights saved to: {INSIGHTS_PATH}")
    print(f"Creatives saved to: {CREATIVES_PATH}")