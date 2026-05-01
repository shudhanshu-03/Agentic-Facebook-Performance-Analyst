import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

# ---------- LLM Helper ----------

def get_openai_client() -> OpenAI:
    """
    Returns an OpenAI client. Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key-for-local")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not base_url and api_key == "dummy-key-for-local":
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
        
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(system_prompt: str, user_prompt: str,
             model: str = None,
             temperature: float = 0.2) -> str:
    """
    Call the OpenAI Chat Completion API and return text content.
    """
    if model is None:
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try to parse JSON. If it fails, attempt to fix common issues.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Simple heuristic cleanup: strip code fences and trailing text
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned = cleaned[first_brace:last_brace + 1]
        return json.loads(cleaned)


# ---------- Data Helpers ----------

def load_facebook_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    # fill ctr if missing
    if "ctr" not in df.columns and {"clicks", "impressions"}.issubset(df.columns):
        df["ctr"] = np.where(df["impressions"] > 0,
                             df["clicks"] / df["impressions"],
                             0.0)
    # fill roas if missing
    if "roas" not in df.columns and {"revenue", "spend"}.issubset(df.columns):
        df["roas"] = np.where(df["spend"] > 0,
                              df["revenue"] / df["spend"],
                              0.0)
    return df


def build_data_summary(df: pd.DataFrame, max_days: int = 30) -> Dict[str, Any]:
    """
    Build a compact JSON-serializable summary of the dataset.
    """
    summary: Dict[str, Any] = {}

    # overall stats
    summary["overall"] = {}
    if "date" in df.columns:
        summary["overall"]["date_min"] = str(df["date"].min().date())
        summary["overall"]["date_max"] = str(df["date"].max().date())

    for col in ["spend", "revenue", "clicks", "impressions", "purchases"]:
        if col in df.columns:
            summary["overall"][f"total_{col}"] = float(df[col].sum())

    if {"revenue", "spend"}.issubset(df.columns):
        total_spend = df["spend"].sum()
        total_rev = df["revenue"].sum()
        summary["overall"]["overall_roas"] = float(total_rev / total_spend) if total_spend > 0 else 0.0

    if {"clicks", "impressions"}.issubset(df.columns):
        total_clicks = df["clicks"].sum()
        total_impr = df["impressions"].sum()
        summary["overall"]["overall_ctr"] = float(total_clicks / total_impr) if total_impr > 0 else 0.0

    # ROAS by date
    if "date" in df.columns and {"spend", "revenue"}.issubset(df.columns):
        g = df.groupby("date").agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
        ).reset_index()
        g["roas"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], 0.0)
        g["ctr"] = np.where(g["impressions"] > 0, g["clicks"] / g["impressions"], 0.0)
        g = g.sort_values("date").tail(max_days)
        summary["roas_by_date"] = [
            {
                "date": str(row["date"].date()),
                "spend": float(row["spend"]),
                "revenue": float(row["revenue"]),
                "roas": float(row["roas"]),
                "ctr": float(row["ctr"]),
            }
            for _, row in g.iterrows()
        ]

    # campaign-level metrics
    if "campaign_name" in df.columns:
        group_cols = ["campaign_name"]
        agg_dict = {}
        for col in ["spend", "revenue", "clicks", "impressions", "purchases"]:
            if col in df.columns:
                agg_dict[col] = "sum"
        g = df.groupby(group_cols).agg(agg_dict).reset_index()
        if "spend" in g.columns and "revenue" in g.columns:
            g["roas"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], 0.0)
        if "impressions" in g.columns and "clicks" in g.columns:
            g["ctr"] = np.where(g["impressions"] > 0, g["clicks"] / g["impressions"], 0.0)

        g = g.sort_values("spend", ascending=False)
        top = g.head(10)
        bottom = g.nsmallest(10, "ctr") if "ctr" in g.columns else g.tail(10)

        def rows_to_dict(df_part: pd.DataFrame):
            return [
                {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                 for k, v in row.items()}
                for row in df_part.to_dict(orient="records")
            ]

        summary["top_campaigns_by_spend"] = rows_to_dict(top)
        summary["low_ctr_campaigns_raw"] = rows_to_dict(bottom)

    # low CTR campaigns with creative info (row level)
    if {"ctr", "campaign_name"}.issubset(df.columns):
        quantile = df["ctr"].quantile(0.25)
        low_ctr_rows = df[df["ctr"] <= quantile].copy()
        low_ctr_rows = low_ctr_rows.sort_values("ctr").head(20)

        summary["low_ctr_campaigns"] = []
        for _, r in low_ctr_rows.iterrows():
            summary["low_ctr_campaigns"].append(
                {
                    "campaign_name": str(r.get("campaign_name", "")),
                    "adset_name": str(r.get("adset_name", "")),
                    "audience_type": str(r.get("audience_type", "")),
                    "platform": str(r.get("platform", "")),
                    "country": str(r.get("country", "")),
                    "creative_type": str(r.get("creative_type", "")),
                    "creative_message": str(r.get("creative_message", "")),
                    "ctr": float(r.get("ctr", 0.0)),
                    "roas": float(r.get("roas", 0.0)),
                    "spend": float(r.get("spend", 0.0)),
                }
            )

    # high CTR creatives
    if "ctr" in df.columns:
        high_ctr_rows = df.sort_values("ctr", ascending=False).head(30)
        summary["high_ctr_creatives"] = []
        for _, r in high_ctr_rows.iterrows():
            summary["high_ctr_creatives"].append(
                {
                    "campaign_name": str(r.get("campaign_name", "")),
                    "creative_type": str(r.get("creative_type", "")),
                    "creative_message": str(r.get("creative_message", "")),
                    "audience_type": str(r.get("audience_type", "")),
                    "ctr": float(r.get("ctr", 0.0)),
                    "roas": float(r.get("roas", 0.0)),
                    "spend": float(r.get("spend", 0.0)),
                }
            )

    return summary


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_log(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
