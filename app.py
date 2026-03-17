import io
import os
import json
import calendar
from datetime import date
from typing import Optional, Any

import pandas as pd
import streamlit as st
from openai import OpenAI

from ads_optimizer_ingestion import AdsOptimizerEngine

st.set_page_config(
    page_title="Evolved Commerce Amazon Ads Command Center",
    page_icon="📈",
    layout="wide",
)


# =========================================================
# Helpers
# =========================================================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Output")
    return output.getvalue()


def safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def get_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def get_uploaded_bytes(file_obj):
    if file_obj is None:
        return None
    return file_obj.getvalue()


def bytes_to_buffer(file_bytes):
    if file_bytes is None:
        return None
    return io.BytesIO(file_bytes)


def load_logo_path() -> Optional[str]:
    possible_paths = [
        "assets/ec_logo.png",
        "assets/ec_logo.jpg",
        "assets/ec_logo.jpeg",
        "assets/logo.png",
        "assets/logo.jpg",
        "ec_logo.png",
        "logo.png",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def upload_status_line(file_obj, success_text: str) -> None:
    if file_obj is not None:
        st.success(success_text)


def render_metric_card(label: str, value: str, tone: str = "brand", small: bool = False) -> None:
    value_class = "metric-value small" if small else "metric-value"
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-label">{label}</div>
            <div class="{value_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_openai_api_key() -> Optional[str]:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_openai_client() -> Optional[OpenAI]:
    api_key = get_openai_api_key()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_openai_model() -> str:
    try:
        if "OPENAI_MODEL" in st.secrets and st.secrets["OPENAI_MODEL"]:
            return st.secrets["OPENAI_MODEL"]
    except Exception:
        pass
    return os.getenv("OPENAI_MODEL", "gpt-4o")


def normalize_match_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_term_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def get_strategy_parameters(strategy_mode: str) -> dict:
    mode = str(strategy_mode).strip().lower()

    if mode == "conservative":
        return {
            "max_bid_up": 0.05,
            "max_bid_down": 0.10,
            "budget_up_pct": 0.05,
            "budget_down_pct": 0.08,
        }
    if mode == "aggressive":
        return {
            "max_bid_up": 0.20,
            "max_bid_down": 0.25,
            "budget_up_pct": 0.15,
            "budget_down_pct": 0.12,
        }

    return {
        "max_bid_up": 0.10,
        "max_bid_down": 0.15,
        "budget_up_pct": 0.10,
        "budget_down_pct": 0.10,
    }


def score_action_confidence(row: dict) -> str:
    clicks = float(row.get("clicks", 0) or 0)
    orders = float(row.get("orders", 0) or 0)
    roas = float(row.get("roas", 0) or 0)
    spend = float(row.get("spend", 0) or 0)
    source_type = str(row.get("source_type", "")).lower()
    action = str(row.get("optimizer_action", "")).upper()

    if clicks >= 20 and orders == 0 and spend >= 20:
        return "HIGH"

    if orders >= 3 and roas >= 3:
        return "HIGH"

    if source_type in {"search_harvest", "negative_keyword"} and clicks < 12:
        return "LOW"

    if source_type == "budget" and clicks < 25:
        return "LOW"

    if source_type == "bid" and clicks < 8:
        return "LOW"

    if 2.5 <= roas <= 4:
        return "LOW"

    if action in {"INCREASE_BID", "DECREASE_BID", "INCREASE_BUDGET", "DECREASE_BUDGET"}:
        return "MEDIUM"

    return "LOW"


def build_narrative(
    account_health: dict,
    simulation_summary: dict,
    enable_monthly_budget_control: bool,
    pacing_status: str,
) -> list[str]:
    notes = []

    health_status = account_health.get("health_status", "unknown")
    account_roas = account_health.get("account_roas", 0)
    adjusted_min_roas = account_health.get("adjusted_min_roas", 0)
    tacos_pct = account_health.get("tacos_pct", None)

    if health_status == "under_target":
        notes.append(
            f"Account ROAS is below target at {account_roas}. The optimizer tightened efficiency controls and used an adjusted ROAS threshold of {adjusted_min_roas}."
        )
    elif health_status == "above_target":
        notes.append(
            f"Account ROAS is healthy at {account_roas}. The optimizer allowed more room for controlled scaling."
        )
    elif health_status == "tacos_constrained":
        notes.append(
            "TACOS guardrails are active. The optimizer tightened scaling behavior to protect total account efficiency."
        )
    else:
        notes.append(
            f"Account health is stable. The optimizer used a balanced rule set around the target ROAS of {adjusted_min_roas}."
        )

    if enable_monthly_budget_control:
        if pacing_status == "Over Pace":
            notes.append(
                "Monthly pacing is currently over target. Budget increases and bid increases were suppressed to help protect the monthly spend cap."
            )
        elif pacing_status == "On Pace":
            notes.append(
                "Monthly pacing is on target. Standard optimization behavior was allowed without pacing suppression."
            )

    if simulation_summary.get("bid_decreases", 0) > 0:
        notes.append(
            f"{simulation_summary['bid_decreases']} bid decreases were generated to reduce waste and tighten weak targets."
        )

    if simulation_summary.get("bid_increases", 0) > 0:
        notes.append(
            f"{simulation_summary['bid_increases']} bid increases were generated for strong-performing targets with scaling headroom."
        )

    if simulation_summary.get("harvested_keywords", 0) > 0:
        notes.append(
            f"{simulation_summary['harvested_keywords']} search terms were harvested into Exact keywords based on conversion and efficiency thresholds."
        )

    if simulation_summary.get("negatives_added", 0) > 0:
        notes.append(
            f"{simulation_summary['negatives_added']} negative phrases were added to reduce wasted clicks from unproductive traffic."
        )

    if simulation_summary.get("budget_increases", 0) > 0 or simulation_summary.get("budget_decreases", 0) > 0:
        notes.append(
            f"Campaign budget actions included {simulation_summary.get('budget_increases', 0)} increases and {simulation_summary.get('budget_decreases', 0)} decreases."
        )

    if tacos_pct is not None:
        notes.append(f"Current TACOS reading from this run is {tacos_pct}%.")

    return notes


# =========================================================
# AI Override Layer
# =========================================================
@st.cache_data(show_spinner=False, ttl=900)
def run_ai_review_cached(payload_json: str, model: str) -> dict:
    client = get_openai_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY not found.")

    instructions = """
You are an expert Amazon Sponsored Products optimizer.

You are reviewing ONLY low-confidence optimization actions. Do not review high-confidence actions.

Your goal:
- KEEP actions that still look directionally correct.
- MODIFY actions only when there is a clear safer alternative.
- REMOVE actions that look too risky or unsupported by data.

Rules:
- Be conservative.
- Prefer KEEP over MODIFY.
- Prefer REMOVE over risky scaling.
- Do not invent data.
- For bid actions, new_action may only be: INCREASE_BID, DECREASE_BID, or NO_ACTION.
- For budget actions, new_action may only be: INCREASE_BUDGET, DECREASE_BUDGET, or NO_ACTION.
- For harvest actions, new_action may only be: HARVEST_TO_EXACT or NO_ACTION.
- For negative keyword actions, new_action may only be: ADD_NEGATIVE_PHRASE or NO_ACTION.
- If decision is KEEP, set new_action to an empty string.
- If decision is REMOVE, set new_action to an empty string.
- If decision is MODIFY, you must supply new_action.
- Use SQP opportunities only as strategy context, not as direct execution instructions.

Return structured JSON only.
"""

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=payload_json,
        max_output_tokens=1400,
        text={
            "format": {
                "type": "json_schema",
                "name": "ai_overrides",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "executive_summary": {"type": "string"},
                        "overrides": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "id": {"type": "string"},
                                    "decision": {
                                        "type": "string",
                                        "enum": ["KEEP", "MODIFY", "REMOVE"]
                                    },
                                    "new_action": {"type": "string"},
                                    "reason": {"type": "string"}
                                },
                                "required": ["id", "decision", "new_action", "reason"]
                            }
                        }
                    },
                    "required": ["executive_summary", "overrides"]
                }
            }
        }
    )

    raw_text = (response.output_text or "").strip()

    if not raw_text:
        raise ValueError("OpenAI returned empty response.")

    return json.loads(raw_text)


def build_ai_action_candidates(
    combined_bulk_updates: pd.DataFrame,
    bid_recommendations: pd.DataFrame,
    search_term_actions: pd.DataFrame,
    campaign_budget_actions: pd.DataFrame,
) -> pd.DataFrame:
    if combined_bulk_updates.empty:
        return pd.DataFrame()

    combined = combined_bulk_updates.copy()
    combined["id"] = combined.index.astype(str)

    bid = bid_recommendations.copy() if not bid_recommendations.empty else pd.DataFrame()
    search = search_term_actions.copy() if not search_term_actions.empty else pd.DataFrame()
    budget = campaign_budget_actions.copy() if not campaign_budget_actions.empty else pd.DataFrame()

    if not bid.empty:
        bid["_campaign_key"] = bid["campaign_name"].map(normalize_match_text)
        bid["_ad_group_key"] = bid["ad_group_name"].map(normalize_match_text)
        bid["_term_key"] = bid["target"].map(normalize_term_text)
        bid["_match_key"] = bid["match_type"].map(normalize_match_text)

    if not search.empty:
        search["_campaign_key"] = search["campaign_name"].map(normalize_match_text)
        search["_ad_group_key"] = search["ad_group_name"].map(normalize_match_text)
        search["_term_key"] = search["search_term"].map(normalize_term_text)
        search["_action_key"] = search["search_term_action"].astype(str).str.upper()

    if not budget.empty:
        budget["_campaign_key"] = budget["campaign_name"].map(normalize_match_text)
        budget["_action_key"] = budget["campaign_action"].astype(str).str.upper()

    rows = []

    for _, row in combined.iterrows():
        candidate = {
            "id": str(row.get("id", "")),
            "source_type": "unknown",
            "optimizer_action": str(row.get("Optimizer Action", "")),
            "campaign_name": str(row.get("Campaign Name", "")),
            "ad_group_name": str(row.get("Ad Group Name", "")),
            "keyword_text": str(row.get("Keyword Text", "")),
            "match_type": str(row.get("Match Type", "")),
            "entity": str(row.get("Entity", "")),
            "clicks": 0.0,
            "orders": 0.0,
            "spend": 0.0,
            "sales": 0.0,
            "roas": 0.0,
            "impression_share_pct": 0.0,
            "current_bid": None,
            "recommended_bid": None,
            "current_daily_budget": None,
            "recommended_daily_budget": None,
            "existing_action": str(row.get("Optimizer Action", "")),
        }

        campaign_key = normalize_match_text(row.get("Campaign Name", ""))
        ad_group_key = normalize_match_text(row.get("Ad Group Name", ""))
        term_key = normalize_term_text(row.get("Keyword Text", ""))
        match_key = normalize_match_text(row.get("Match Type", ""))
        optimizer_action = str(row.get("Optimizer Action", "")).upper()
        entity = str(row.get("Entity", ""))

        if entity == "Keyword" and optimizer_action in {"INCREASE_BID", "DECREASE_BID"} and not bid.empty:
            match = bid[
                (bid["_campaign_key"] == campaign_key)
                & (bid["_ad_group_key"] == ad_group_key)
                & (bid["_term_key"] == term_key)
                & (bid["_match_key"] == match_key)
                & (bid["recommended_action"].astype(str).str.upper() == optimizer_action)
            ]

            if not match.empty:
                m = match.iloc[0]
                candidate.update(
                    {
                        "source_type": "bid",
                        "clicks": get_number(m.get("clicks")),
                        "orders": get_number(m.get("orders")),
                        "spend": get_number(m.get("spend")),
                        "sales": get_number(m.get("sales")),
                        "roas": get_number(m.get("roas")),
                        "impression_share_pct": get_number(m.get("impression_share_pct")),
                        "current_bid": get_number(m.get("current_bid")),
                        "recommended_bid": get_number(m.get("recommended_bid")),
                    }
                )

        elif optimizer_action in {"HARVEST_TO_EXACT", "ADD_NEGATIVE_PHRASE"} and not search.empty:
            match = search[
                (search["_campaign_key"] == campaign_key)
                & (search["_ad_group_key"] == ad_group_key)
                & (search["_term_key"] == term_key)
                & (search["_action_key"] == optimizer_action)
            ]

            if not match.empty:
                m = match.iloc[0]
                source_type = "search_harvest" if optimizer_action == "HARVEST_TO_EXACT" else "negative_keyword"
                candidate.update(
                    {
                        "source_type": source_type,
                        "clicks": get_number(m.get("clicks")),
                        "orders": get_number(m.get("orders")),
                        "spend": get_number(m.get("spend")),
                        "sales": get_number(m.get("sales")),
                        "roas": get_number(m.get("roas")),
                        "impression_share_pct": 0.0,
                    }
                )

        elif entity == "Campaign" and optimizer_action in {"INCREASE_BUDGET", "DECREASE_BUDGET"} and not budget.empty:
            match = budget[
                (budget["_campaign_key"] == campaign_key)
                & (budget["_action_key"] == optimizer_action)
            ]

            if not match.empty:
                m = match.iloc[0]
                candidate.update(
                    {
                        "source_type": "budget",
                        "clicks": get_number(m.get("clicks")),
                        "orders": get_number(m.get("orders")),
                        "spend": get_number(m.get("spend")),
                        "sales": get_number(m.get("sales")),
                        "roas": get_number(m.get("roas")),
                        "impression_share_pct": get_number(m.get("avg_impression_share_pct")),
                        "current_daily_budget": get_number(m.get("daily_budget")),
                        "recommended_daily_budget": get_number(m.get("recommended_daily_budget")),
                    }
                )

        candidate["confidence"] = score_action_confidence(candidate)
        rows.append(candidate)

    return pd.DataFrame(rows)


def build_ai_override_payload(
    low_conf_df: pd.DataFrame,
    account_summary: dict,
    account_health: dict,
    simulation_summary: dict,
    sqp_opportunities: pd.DataFrame,
    sqp_summary: dict,
) -> dict:
    keep_cols = [
        "id",
        "source_type",
        "optimizer_action",
        "campaign_name",
        "ad_group_name",
        "keyword_text",
        "match_type",
        "clicks",
        "orders",
        "spend",
        "sales",
        "roas",
        "impression_share_pct",
        "current_bid",
        "recommended_bid",
        "current_daily_budget",
        "recommended_daily_budget",
        "confidence",
    ]
    cols = [c for c in keep_cols if c in low_conf_df.columns]

    sqp_context = []
    if sqp_opportunities is not None and not sqp_opportunities.empty:
        keep_sqp_cols = [
            c for c in [
                "search_query",
                "search_query_score",
                "search_query_volume",
                "purchase_rate_pct",
                "purchases_total_count",
                "purchases_brand_share_pct",
                "opportunity_tier",
                "recommended_action",
                "in_search_term_report",
            ] if c in sqp_opportunities.columns
        ]
        sqp_context = sqp_opportunities.head(10)[keep_sqp_cols].to_dict(orient="records")

    return {
        "account_summary": account_summary,
        "account_health": account_health,
        "simulation_summary": simulation_summary,
        "sqp_summary": sqp_summary or {},
        "sqp_top_opportunities": sqp_context,
        "low_confidence_actions": low_conf_df[cols].to_dict(orient="records"),
    }


def apply_ai_overrides_to_combined(
    combined_bulk_updates: pd.DataFrame,
    ai_response: dict,
    ai_candidates_df: pd.DataFrame,
    strategy_mode: str,
    max_bid_cap: float,
    max_budget_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if combined_bulk_updates.empty:
        return combined_bulk_updates.copy(), pd.DataFrame()

    strategy = get_strategy_parameters(strategy_mode)
    override_map = {str(o.get("id")): o for o in ai_response.get("overrides", [])}
    candidate_map = {
        str(row["id"]): row.to_dict()
        for _, row in ai_candidates_df.iterrows()
        if "id" in ai_candidates_df.columns
    }

    updated_rows = []
    log_rows = []

    for idx, row in combined_bulk_updates.copy().iterrows():
        row_dict = row.to_dict()
        rid = str(idx)

        original_action = str(row_dict.get("Optimizer Action", ""))
        final_action = original_action
        decision = "KEEP"
        reason = ""
        source_type = candidate_map.get(rid, {}).get("source_type", "unknown")
        was_removed = False

        override = override_map.get(rid)

        if override:
            decision = str(override.get("decision", "KEEP")).upper()
            reason = str(override.get("reason", "")).strip()
            new_action = str(override.get("new_action", "")).upper().strip()

            if decision == "REMOVE" or new_action == "NO_ACTION":
                was_removed = True

            elif decision == "MODIFY":
                candidate = candidate_map.get(rid, {})
                current_bid = get_number(candidate.get("current_bid"))
                current_budget = get_number(candidate.get("current_daily_budget"))

                if source_type == "bid" and new_action in {"INCREASE_BID", "DECREASE_BID"} and current_bid > 0:
                    final_action = new_action
                    if new_action == "INCREASE_BID":
                        new_bid = round(min(current_bid * (1 + strategy["max_bid_up"]), max_bid_cap), 2)
                    else:
                        new_bid = round(max(current_bid * (1 - strategy["max_bid_down"]), 0.02), 2)
                    row_dict["Bid"] = new_bid

                elif source_type == "budget" and new_action in {"INCREASE_BUDGET", "DECREASE_BUDGET"} and current_budget > 0:
                    final_action = new_action
                    if new_action == "INCREASE_BUDGET":
                        new_budget = round(min(current_budget * (1 + strategy["budget_up_pct"]), max_budget_cap), 2)
                    else:
                        new_budget = round(max(current_budget * (1 - strategy["budget_down_pct"]), 1.00), 2)
                    row_dict["Daily Budget"] = new_budget

                elif source_type == "search_harvest" and new_action == "HARVEST_TO_EXACT":
                    final_action = new_action

                elif source_type == "negative_keyword" and new_action == "ADD_NEGATIVE_PHRASE":
                    final_action = new_action

                else:
                    final_action = original_action

        if not was_removed:
            row_dict["Optimizer Action"] = final_action
            updated_rows.append(row_dict)

        candidate = candidate_map.get(rid, {})
        log_rows.append(
            {
                "ID": rid,
                "Source Type": source_type,
                "Campaign Name": candidate.get("campaign_name", row_dict.get("Campaign Name", "")),
                "Ad Group Name": candidate.get("ad_group_name", row_dict.get("Ad Group Name", "")),
                "Keyword Text": candidate.get("keyword_text", row_dict.get("Keyword Text", "")),
                "Confidence": candidate.get("confidence", ""),
                "Clicks": candidate.get("clicks", 0),
                "Orders": candidate.get("orders", 0),
                "Spend": candidate.get("spend", 0),
                "Sales": candidate.get("sales", 0),
                "ROAS": candidate.get("roas", 0),
                "Original Action": original_action,
                "Decision": decision,
                "Final Action": "REMOVED" if was_removed else final_action,
                "Reason": reason,
            }
        )

    updated_df = pd.DataFrame(updated_rows)
    log_df = pd.DataFrame(log_rows)

    return updated_df, log_df


def build_ai_impact_summary(
    ai_candidates_df: pd.DataFrame,
    ai_override_log_df: pd.DataFrame,
    original_count: int,
    final_count: int,
    executive_summary: str,
) -> dict:
    if ai_override_log_df.empty:
        return {
            "reviewed": 0,
            "kept": 0,
            "modified": 0,
            "removed": 0,
            "removed_spend": 0.0,
            "removed_sales": 0.0,
            "original_count": original_count,
            "final_count": final_count,
            "executive_summary": executive_summary,
        }

    reviewed = len(ai_override_log_df)
    kept = int((ai_override_log_df["Decision"] == "KEEP").sum())
    modified = int((ai_override_log_df["Decision"] == "MODIFY").sum())
    removed = int((ai_override_log_df["Final Action"] == "REMOVED").sum())

    removed_rows = ai_override_log_df[ai_override_log_df["Final Action"] == "REMOVED"]

    return {
        "reviewed": reviewed,
        "kept": kept,
        "modified": modified,
        "removed": removed,
        "removed_spend": round(float(pd.to_numeric(removed_rows["Spend"], errors="coerce").fillna(0).sum()), 2),
        "removed_sales": round(float(pd.to_numeric(removed_rows["Sales"], errors="coerce").fillna(0).sum()), 2),
        "original_count": original_count,
        "final_count": final_count,
        "executive_summary": executive_summary,
    }


# =========================================================
# Styles
# =========================================================
st.markdown(
    """
    <style>
        .main > div {
            padding-top: 1.1rem;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1.75rem;
            max-width: 1440px;
        }

        [data-testid="stSidebar"] {
            background: #F3F4F6;
        }

        .brand-shell {
            background: linear-gradient(135deg, #EA580C 0%, #1F2937 100%);
            border-radius: 20px;
            padding: 65px 30px;
            color: white;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
        }

        .brand-title {
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0;
        }

        .brand-subtitle {
            font-size: 1rem;
            opacity: 0.96;
            margin-top: 0.55rem;
        }

        .section-title {
            font-size: 1.18rem;
            font-weight: 750;
            color: #111827;
            margin-bottom: 0.2rem;
        }

        .section-note {
            color: #4B5563;
            font-size: 0.94rem;
            margin-bottom: 0.9rem;
        }

        .section-divider {
            border-top: 1px solid #E5E7EB;
            margin: 1.1rem 0 1.15rem 0;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 600;
            border: none;
            transition: all 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
        }

        details {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 0.35rem 0.75rem;
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
        }

        summary {
            font-weight: 700;
            color: #1F2937;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            min-height: 80px;
        }

        .metric-label {
            font-size: 0.75rem;
            color: #6B7280;
            margin-bottom: 0.25rem;
            font-weight: 600;
        }

        .metric-value {
            font-size: 1.5rem;
            line-height: 1.1;
            font-weight: 800;
            color: #1F2937;
            word-break: break-word;
        }

        .metric-value.small {
            font-size: 1.15rem;
        }

        .metric-card.good {
            border-left: 4px solid #10B981;
        }

        .metric-card.warn {
            border-left: 4px solid #F59E0B;
        }

        .metric-card.bad {
            border-left: 4px solid #EF4444;
        }

        .metric-card.brand {
            border-left: 4px solid #F47322;
        }

        .footer-note {
            color: #6B7280;
            font-size: 0.85rem;
            text-align: center;
            opacity: 0.9;
            margin-top: 0.35rem;
            margin-bottom: 0.5rem;
        }

        .footer-note::before {
            content: "";
            display: block;
            width: 120px;
            height: 1px;
            background: #E5E7EB;
            margin: 0 auto 6px auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("## Tool Guide")
    st.markdown(
        """
**What this tool does**
- Adjusts bids
- Harvests winning search terms into Exact matches
- Adds negative phrases for wasted traffic
- Adjusts campaign daily budgets
- Optionally enforces TACOS and monthly pacing guardrails
- Optionally uses prior-month SQP Simple View for keyword opportunity context
"""
    )

    st.markdown("---")
    st.markdown("## Required uploads")
    st.markdown(
        """
1. Bulk Sheet  
2. Search Term Report  
3. Targeting Report  
4. Impression Share Report
"""
    )

    st.markdown("## Optional uploads")
    st.markdown(
        """
- Seller Central Business Report for TACOS control  
- Search Query Performance Report (**Simple View**, **prior month only**)
"""
    )

    st.markdown("---")
    st.markdown("## Recommended Workflow")
    st.markdown(
        """
- Start with **Balanced**
- Use **Losing KW Action = Both**
- Use monthly pacing only for budget-sensitive clients
- Upload prior-month SQP only in **Simple View**
- Review diagnostics before running
- Review output tables before download
"""
    )

    st.markdown("---")
    st.markdown("## AI Optimization Layer")

    enable_ai_review = st.checkbox("Enable AI Optimization Layer", value=True)

    if enable_ai_review:
        st.caption("AI automatically reviews low-confidence actions and improves the final output.")
        st.caption(f"AI Model: {get_openai_model()}")

    api_key_present = bool(get_openai_api_key())
    if enable_ai_review and not api_key_present:
        st.warning("OPENAI_API_KEY not found. Add it to environment variables or Streamlit secrets.")

    st.markdown("---")
    st.markdown("**Version:** 1.3")
    st.markdown("**Owner:** Jake Yorgason, Evolved Commerce")


# =========================================================
# Header
# =========================================================
logo_path = load_logo_path()

header_left, header_right = st.columns([1.35, 5.65], gap="medium")

with header_left:
    if logo_path:
        st.image(logo_path, width=500)

with header_right:
    st.markdown(
        """
        <div class="brand-shell">
            <div class="brand-title">Evolved Commerce<br>Amazon Ads Command Center</div>
            <div class="brand-subtitle">
                Bid optimization • Search harvesting • Negative mining • Budget pacing • TACOS control • SQP context • AI optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Settings
# =========================================================
st.markdown("<!-- redeploy -->", unsafe_allow_html=True)

st.markdown('<div class="section-title">Optimization Settings</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Core controls for bidding, efficiency, and zero-order action logic.</div>',
    unsafe_allow_html=True,
)

r1c1, r1c2, r1c3 = st.columns(3)

with r1c1:
    min_roas = st.number_input(
        "Minimum ROAS Target",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
    )

with r1c2:
    min_clicks = st.number_input(
        "Minimum Clicks Before Standard Efficiency Actions",
        min_value=1,
        max_value=100,
        value=8,
        step=1,
        help="This threshold controls standard bid decreases for weak efficiency.",
    )

with r1c3:
    strategy_mode = st.selectbox(
        "Strategy Mode",
        options=["Conservative", "Balanced", "Aggressive"],
        index=1,
    )

r2c1, r2c2 = st.columns(2)

with r2c1:
    losing_kw_click_threshold = st.number_input(
        "Losing KW Click Threshold",
        min_value=1,
        max_value=100,
        value=12,
        step=1,
        help="After this many clicks and 0 orders, the app will take the selected Losing KW Action.",
    )

with r2c2:
    losing_kw_action = st.selectbox(
        "Losing KW Action Type",
        options=["Decrease Bid", "Add Negative", "Both", "None"],
        index=2,
        help="Choose exactly what the app should do when a keyword or search term reaches the losing threshold.",
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Optimization Actions</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Enable or disable the main optimization behaviors for this run.</div>',
    unsafe_allow_html=True,
)

core1, core2, core3, core4 = st.columns(4)

with core1:
    enable_bid_updates = st.checkbox("Bid Updates", value=True)

with core2:
    enable_search_harvesting = st.checkbox("Search Harvesting", value=True)

with core3:
    enable_negative_keywords = st.checkbox("Negative Keywords", value=True)

with core4:
    enable_budget_updates = st.checkbox("Budget Updates", value=True)

with st.expander("Efficiency Guardrails", expanded=False):
    eg1, eg2, eg3 = st.columns([1, 1, 1.2])

    with eg1:
        enable_tacos_control = st.checkbox("Enable TACOS Control", value=False)

    with eg2:
        max_tacos_target = st.number_input(
            "Maximum TACOS %",
            min_value=1.0,
            max_value=100.0,
            value=15.0,
            step=0.5,
            disabled=not enable_tacos_control,
        )

    with eg3:
        st.caption("Use Seller Central business report when TACOS control is enabled.")

with st.expander("Budget Guardrails", expanded=False):
    bg1, bg2 = st.columns(2)

    with bg1:
        enable_monthly_budget_control = st.checkbox(
            "Enable Monthly Budget Control",
            value=False,
        )

    with bg2:
        pacing_buffer_pct = st.number_input(
            "Pacing Buffer %",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            disabled=not enable_monthly_budget_control,
        )

    bg3, bg4 = st.columns(2)

    with bg3:
        monthly_account_budget = st.number_input(
            "Monthly Account Budget",
            min_value=0.0,
            value=0.0,
            step=100.0,
            disabled=not enable_monthly_budget_control,
        )

    with bg4:
        month_to_date_spend = st.number_input(
            "Month-to-Date Ad Spend",
            min_value=0.0,
            value=0.0,
            step=10.0,
            disabled=not enable_monthly_budget_control,
        )

    bg5, bg6 = st.columns(2)

    with bg5:
        max_bid_cap = st.number_input(
            "Maximum Recommended Bid",
            min_value=0.05,
            max_value=20.0,
            value=5.0,
            step=0.05,
        )

    with bg6:
        max_budget_cap = st.number_input(
            "Maximum Recommended Daily Budget",
            min_value=1.0,
            max_value=10000.0,
            value=500.0,
            step=1.0,
        )


# =========================================================
# Pacing
# =========================================================
if enable_monthly_budget_control and monthly_account_budget > 0:
    today = date.today()
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    remaining_days = max(days_in_month - today.day + 1, 1)
    remaining_budget = monthly_account_budget - month_to_date_spend
    allowed_daily_pace = remaining_budget / remaining_days if remaining_days > 0 else 0
    current_daily_pace = month_to_date_spend / max(today.day, 1)
    pacing_limit = allowed_daily_pace * (1 + pacing_buffer_pct / 100)
    pacing_status = "Over Pace" if current_daily_pace > pacing_limit else "On Pace"

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric("Remaining Budget", f"${remaining_budget:,.2f}")
    with p2:
        st.metric("Remaining Days", remaining_days)
    with p3:
        st.metric("Allowed Daily Pace", f"${allowed_daily_pace:,.2f}")
    with p4:
        st.metric("Pacing Status", pacing_status)
else:
    pacing_status = "Not Enabled"


# =========================================================
# Uploads
# =========================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Upload Amazon Reports</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Upload the report set to unlock diagnostics, SQP context, and optimization preview.</div>',
    unsafe_allow_html=True,
)

up1, up2 = st.columns(2)

with up1:
    bulk_file = st.file_uploader("Bulk Sheet", type=["xlsx"])
    upload_status_line(bulk_file, "Bulk Sheet uploaded")

    search_file = st.file_uploader("Search Term Report", type=["xlsx"])
    upload_status_line(search_file, "Search Term Report uploaded")

    targeting_file = st.file_uploader("Targeting Report", type=["xlsx"])
    upload_status_line(targeting_file, "Targeting Report uploaded")

with up2:
    impression_file = st.file_uploader("Impression Share Report", type=["csv"])
    upload_status_line(impression_file, "Impression Share Report uploaded")

    business_file = st.file_uploader(
        "Seller Central Business Report (optional unless TACOS control is enabled)",
        type=["xlsx", "csv"],
    )
    if business_file is not None:
        st.success("Seller Central Business Report uploaded")

    sqp_file = st.file_uploader(
        "Search Query Performance Report (optional — prior month, Simple View only)",
        type=["csv"],
        help="Use the prior month's SQP report in Simple View only.",
    )
    if sqp_file is not None:
        st.success("Search Query Performance Report uploaded")

bulk_bytes = get_uploaded_bytes(bulk_file)
search_bytes = get_uploaded_bytes(search_file)
targeting_bytes = get_uploaded_bytes(targeting_file)
impression_bytes = get_uploaded_bytes(impression_file)
business_bytes = get_uploaded_bytes(business_file)
sqp_bytes = get_uploaded_bytes(sqp_file)

required_ready = all([bulk_bytes, search_bytes, targeting_bytes, impression_bytes])
tacos_ready = (not enable_tacos_control) or (business_bytes is not None)


def build_engine() -> AdsOptimizerEngine:
    return AdsOptimizerEngine(
        bulk_file=bytes_to_buffer(bulk_bytes),
        search_term_file=bytes_to_buffer(search_bytes),
        targeting_file=bytes_to_buffer(targeting_bytes),
        impression_share_file=bytes_to_buffer(impression_bytes),
        business_report_file=bytes_to_buffer(business_bytes),
        sqp_report_file=bytes_to_buffer(sqp_bytes),
        min_roas=min_roas,
        min_clicks=min_clicks,
        zero_order_click_threshold=losing_kw_click_threshold,
        zero_order_action=losing_kw_action,
        strategy_mode=strategy_mode,
        enable_bid_updates=enable_bid_updates,
        enable_search_harvesting=enable_search_harvesting,
        enable_negative_keywords=enable_negative_keywords,
        enable_budget_updates=enable_budget_updates,
        enable_tacos_control=enable_tacos_control,
        max_tacos_target=max_tacos_target,
        enable_monthly_budget_control=enable_monthly_budget_control,
        monthly_account_budget=monthly_account_budget,
        month_to_date_spend=month_to_date_spend,
        pacing_buffer_pct=pacing_buffer_pct,
        max_bid_cap=max_bid_cap,
        max_budget_cap=max_budget_cap,
    )


# =========================================================
# Diagnostics
# =========================================================
diagnostics = None
account_health = {}
account_summary = {}
preview = {}
smart_warnings = []
optimization_suggestions = []
campaign_health_dashboard = pd.DataFrame()
sqp_opportunities = pd.DataFrame()
sqp_summary = {}

if required_ready and tacos_ready:
    try:
        with st.spinner("Reading account diagnostics..."):
            diagnostics = build_engine().analyze()

        diagnostics = safe_dict(diagnostics)
        account_health = safe_dict(diagnostics.get("account_health"))
        account_summary = safe_dict(diagnostics.get("account_summary"))
        preview = safe_dict(diagnostics.get("pre_run_preview"))
        smart_warnings = safe_list(diagnostics.get("smart_warnings"))
        optimization_suggestions = safe_list(diagnostics.get("optimization_suggestions"))
        campaign_health_dashboard = safe_df(diagnostics.get("campaign_health_dashboard"))
        sqp_opportunities = safe_df(diagnostics.get("sqp_opportunities"))
        sqp_summary = safe_dict(diagnostics.get("sqp_summary"))

    except Exception as e:
        st.error(f"Diagnostics failed: {e}")
        diagnostics = None

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if diagnostics is not None:
    st.markdown('<div class="section-title">Campaign Health Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">A quick health read before generating any changes.</div>',
        unsafe_allow_html=True,
    )

    tacos_pct = account_health.get("tacos_pct")
    account_roas = get_number(account_health.get("account_roas"))
    adjusted_min_roas = get_number(account_health.get("adjusted_min_roas"), min_roas)

    tacos_display = (
        f"{tacos_pct}%"
        if tacos_pct is not None
        else ("Disabled" if not enable_tacos_control else "Missing")
    )

    dh1, dh2, dh3, dh4, dh5, dh6 = st.columns(6)

    with dh1:
        render_metric_card("Ad Spend", f"${get_number(account_summary.get('total_spend')):,.2f}", tone="brand")
    with dh2:
        render_metric_card("Ad Sales", f"${get_number(account_summary.get('total_sales')):,.2f}", tone="brand")
    with dh3:
        roas_tone = "good" if account_roas >= adjusted_min_roas else "bad"
        render_metric_card("Account ROAS", f"{account_roas:.2f}", tone=roas_tone)
    with dh4:
        render_metric_card("TACOS", str(tacos_display), tone="warn", small=True)
    with dh5:
        render_metric_card("Under Target", str(get_int(account_summary.get("campaigns_under_target"))), tone="bad")
    with dh6:
        render_metric_card("Scalable", str(get_int(account_summary.get("campaigns_scalable"))), tone="good")

    if sqp_summary.get("uploaded"):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">SQP Opportunity Snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Optional prior-month SQP Simple View insights used for keyword opportunity context.</div>',
            unsafe_allow_html=True,
        )

        sq1, sq2, sq3, sq4 = st.columns(4)

        with sq1:
            render_metric_card("High Opportunity", str(get_int(sqp_summary.get("high_opportunity"))), tone="good")
        with sq2:
            render_metric_card("Monitor", str(get_int(sqp_summary.get("monitor"))), tone="warn")
        with sq3:
            render_metric_card("Total SQP Queries", str(get_int(sqp_summary.get("total_queries"))), tone="brand")
        with sq4:
            render_metric_card("Harvest Overlap", str(get_int(sqp_summary.get("harvest_overlap"))), tone="good")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Smart Warnings & Suggestions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Key things to watch before running the optimizer.</div>',
        unsafe_allow_html=True,
    )

    sw1, sw2 = st.columns(2)

    with sw1:
        st.markdown("**Warnings**")
        if smart_warnings:
            for item in smart_warnings:
                st.markdown(f"- {item}")
        else:
            st.info("No warnings for this diagnostic run.")

    with sw2:
        st.markdown("**Suggestions**")
        if optimization_suggestions:
            for item in optimization_suggestions:
                st.markdown(f"- {item}")
        else:
            st.info("No suggestions generated for this diagnostic run.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Pre-Run Action Preview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Estimated actions based on the current settings and uploaded files.</div>',
        unsafe_allow_html=True,
    )

    pr1, pr2, pr3, pr4, pr5, pr6 = st.columns(6)

    with pr1:
        render_metric_card("Bid Increases", str(get_int(preview.get("bid_increases"))), tone="good")
    with pr2:
        render_metric_card("Bid Decreases", str(get_int(preview.get("bid_decreases"))), tone="warn")
    with pr3:
        render_metric_card("Negatives", str(get_int(preview.get("negatives_added"))), tone="warn")
    with pr4:
        render_metric_card("Harvests", str(get_int(preview.get("harvested_keywords"))), tone="good")
    with pr5:
        render_metric_card("Budget Increases", str(get_int(preview.get("budget_increases"))), tone="good")
    with pr6:
        render_metric_card("Budget Decreases", str(get_int(preview.get("budget_decreases"))), tone="warn")

    with st.expander("Campaign Health Table", expanded=False):
        if not campaign_health_dashboard.empty:
            st.dataframe(campaign_health_dashboard, use_container_width=True)
        else:
            st.info("No campaign health table available.")

    if sqp_summary.get("uploaded"):
        with st.expander("Top SQP Opportunities", expanded=False):
            show_cols = [
                c for c in [
                    "search_query",
                    "search_query_score",
                    "search_query_volume",
                    "purchase_rate_pct",
                    "purchases_total_count",
                    "purchases_brand_share_pct",
                    "opportunity_tier",
                    "recommended_action",
                    "in_search_term_report",
                ] if c in sqp_opportunities.columns
            ]
            if not sqp_opportunities.empty and show_cols:
                st.dataframe(sqp_opportunities[show_cols].head(25), use_container_width=True)
            else:
                st.info("No SQP opportunities available.")

    if enable_ai_review and api_key_present:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.info("AI optimization is enabled. After you run the optimizer, AI will automatically review low-confidence actions and use SQP context if available.")


# =========================================================
# Allowed actions summary
# =========================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What This Run Is Allowed To Do</div>', unsafe_allow_html=True)

if losing_kw_action == "Decrease Bid":
    losing_kw_description = (
        f"If a keyword or target reaches **{losing_kw_click_threshold} clicks and 0 orders**, "
        f"the app will **decrease the bid**. It will **not add a negative keyword** from that rule."
    )
elif losing_kw_action == "Add Negative":
    losing_kw_description = (
        f"If a search term reaches **{losing_kw_click_threshold} clicks and 0 orders**, "
        f"the app will **add a Negative Phrase keyword**. It will **not decrease the bid** from that rule."
    )
elif losing_kw_action == "Both":
    losing_kw_description = (
        f"If performance reaches **{losing_kw_click_threshold} clicks and 0 orders**, "
        f"the app may **decrease the bid** and **add a Negative Phrase keyword**, depending on the data source."
    )
else:
    losing_kw_description = (
        f"If performance reaches **{losing_kw_click_threshold} clicks and 0 orders**, "
        f"the app will **take no Losing KW Action**."
    )

st.markdown(f"- {losing_kw_description}")
st.markdown(
    f"- If a target falls below **ROAS {min_roas:.1f}** and has at least **{min_clicks} clicks**, the app may **decrease the bid**."
)
st.markdown("- If a target materially exceeds your ROAS goal, the app may **increase the bid**.")
st.markdown("- Search term harvesting creates **Exact keywords** at the ad group level.")
st.markdown("- Budget updates apply at the campaign daily budget level, not the account level.")
st.markdown("- Prior-month SQP Simple View, when uploaded, is used only for keyword opportunity context and AI guidance, not direct execution logic.")

if enable_tacos_control:
    st.markdown(
        f"- If account TACOS rises above **{max_tacos_target:.1f}%**, the app will tighten scaling rules."
    )

if enable_monthly_budget_control and monthly_account_budget > 0:
    st.markdown(
        "- Monthly budget control is enabled. If current pacing exceeds the allowed pace for the month, the app will block budget increases and block bid increases."
    )

if enable_ai_review:
    st.markdown("- AI optimization is enabled. It will automatically review only low-confidence actions and refine the final bulk upload output.")


# =========================================================
# Run optimizer
# =========================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

button_col1, button_col2, button_col3 = st.columns([3, 2, 3])
with button_col2:
    run_optimizer = st.button("Run Optimization", type="primary", use_container_width=True)

if run_optimizer:
    if not required_ready:
        st.error("Please upload bulk sheet, search term report, targeting report, and impression share report.")
    elif enable_tacos_control and business_bytes is None:
        st.error("Please upload a Seller Central business report to use TACOS control.")
    else:
        try:
            with st.spinner("Analyzing campaigns and generating optimizations..."):
                st.session_state["last_outputs"] = build_engine().process()
            st.success("Optimization complete.")
        except Exception as e:
            st.error(f"Optimization failed: {e}")


# =========================================================
# Results
# =========================================================
if "last_outputs" in st.session_state:
    outputs = safe_dict(st.session_state["last_outputs"])

    combined_bulk_updates = safe_df(outputs.get("combined_bulk_updates"))
    bid_recommendations = safe_df(outputs.get("bid_recommendations"))
    search_term_actions = safe_df(outputs.get("search_term_actions"))
    campaign_budget_actions = safe_df(outputs.get("campaign_budget_actions"))
    output_account_health = safe_dict(outputs.get("account_health"))
    simulation_summary = safe_dict(outputs.get("simulation_summary"))
    run_history = safe_df(outputs.get("run_history"))
    output_campaign_health_dashboard = safe_df(outputs.get("campaign_health_dashboard"))
    output_smart_warnings = safe_list(outputs.get("smart_warnings"))
    output_optimization_suggestions = safe_list(outputs.get("optimization_suggestions"))
    output_account_summary = safe_dict(outputs.get("account_summary"))
    output_sqp_opportunities = safe_df(outputs.get("sqp_opportunities"))
    output_sqp_summary = safe_dict(outputs.get("sqp_summary"))

    ai_candidates_df = pd.DataFrame()
    ai_override_log_df = pd.DataFrame()
    ai_impact_summary = {}
    ai_executive_summary = ""

    original_bulk_count = len(combined_bulk_updates)

    if enable_ai_review and api_key_present and not combined_bulk_updates.empty:
        try:
            ai_candidates_df = build_ai_action_candidates(
                combined_bulk_updates=combined_bulk_updates,
                bid_recommendations=bid_recommendations,
                search_term_actions=search_term_actions,
                campaign_budget_actions=campaign_budget_actions,
            )

            low_conf_df = ai_candidates_df[ai_candidates_df["confidence"] == "LOW"].copy().head(25)

            if not low_conf_df.empty:
                payload = build_ai_override_payload(
                    low_conf_df=low_conf_df,
                    account_summary=output_account_summary,
                    account_health=output_account_health,
                    simulation_summary=simulation_summary,
                    sqp_opportunities=output_sqp_opportunities,
                    sqp_summary=output_sqp_summary,
                )

                ai_response = run_ai_review_cached(
                    payload_json=json.dumps(payload, default=str),
                    model=get_openai_model(),
                )

                ai_executive_summary = str(ai_response.get("executive_summary", "")).strip()

                combined_bulk_updates, ai_override_log_df = apply_ai_overrides_to_combined(
                    combined_bulk_updates=combined_bulk_updates,
                    ai_response=ai_response,
                    ai_candidates_df=ai_candidates_df,
                    strategy_mode=strategy_mode,
                    max_bid_cap=max_bid_cap,
                    max_budget_cap=max_budget_cap,
                )

                ai_impact_summary = build_ai_impact_summary(
                    ai_candidates_df=low_conf_df,
                    ai_override_log_df=ai_override_log_df,
                    original_count=original_bulk_count,
                    final_count=len(combined_bulk_updates),
                    executive_summary=ai_executive_summary,
                )

        except Exception as e:
            st.warning(f"AI optimization skipped: {e}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Optimization Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">High-level actions generated by this run.</div>',
        unsafe_allow_html=True,
    )

    summary1, summary2, summary3, summary4, summary5, summary6 = st.columns(6)

    with summary1:
        render_metric_card("Bid Increases", str(get_int(simulation_summary.get("bid_increases"))), tone="good")
    with summary2:
        render_metric_card("Bid Decreases", str(get_int(simulation_summary.get("bid_decreases"))), tone="warn")
    with summary3:
        render_metric_card("Negatives", str(get_int(simulation_summary.get("negatives_added"))), tone="warn")
    with summary4:
        render_metric_card("Harvests", str(get_int(simulation_summary.get("harvested_keywords"))), tone="good")
    with summary5:
        render_metric_card("Budget Increases", str(get_int(simulation_summary.get("budget_increases"))), tone="good")
    with summary6:
        render_metric_card("Budget Decreases", str(get_int(simulation_summary.get("budget_decreases"))), tone="warn")

    if output_sqp_summary.get("uploaded"):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">SQP Opportunity Reporting</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Prior-month Simple View SQP insights available for keyword opportunity planning.</div>',
            unsafe_allow_html=True,
        )

        sq1, sq2, sq3, sq4 = st.columns(4)

        with sq1:
            render_metric_card("High Opportunity", str(get_int(output_sqp_summary.get("high_opportunity"))), tone="good")
        with sq2:
            render_metric_card("Monitor", str(get_int(output_sqp_summary.get("monitor"))), tone="warn")
        with sq3:
            render_metric_card("Total SQP Queries", str(get_int(output_sqp_summary.get("total_queries"))), tone="brand")
        with sq4:
            render_metric_card("Harvest Overlap", str(get_int(output_sqp_summary.get("harvest_overlap"))), tone="good")

    if enable_ai_review and ai_impact_summary:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">AI Impact Reporting</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">AI reviewed only low-confidence actions and refined the final bulk output.</div>',
            unsafe_allow_html=True,
        )

        ai1, ai2, ai3, ai4, ai5, ai6 = st.columns(6)

        with ai1:
            render_metric_card("Reviewed", str(get_int(ai_impact_summary.get("reviewed"))), tone="brand")
        with ai2:
            render_metric_card("Modified", str(get_int(ai_impact_summary.get("modified"))), tone="warn")
        with ai3:
            render_metric_card("Removed", str(get_int(ai_impact_summary.get("removed"))), tone="bad")
        with ai4:
            render_metric_card("Final Output Rows", str(get_int(ai_impact_summary.get("final_count"))), tone="good")
        with ai5:
            render_metric_card("Removed Spend", f"${get_number(ai_impact_summary.get('removed_spend')):,.2f}", tone="warn", small=True)
        with ai6:
            render_metric_card("Removed Sales", f"${get_number(ai_impact_summary.get('removed_sales')):,.2f}", tone="warn", small=True)

        if ai_executive_summary:
            st.markdown(f"**AI Summary:** {ai_executive_summary}")

        with st.expander("AI Override Log", expanded=False):
            if not ai_override_log_df.empty:
                st.dataframe(ai_override_log_df, use_container_width=True)
            else:
                st.info("No AI override actions were recorded.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    narrative_points = build_narrative(
        account_health=output_account_health,
        simulation_summary=simulation_summary,
        enable_monthly_budget_control=enable_monthly_budget_control,
        pacing_status=pacing_status,
    )

    st.markdown('<div class="section-title">Optimization Narrative</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Plain-English interpretation of what the optimizer did and why.</div>',
        unsafe_allow_html=True,
    )

    for point in narrative_points:
        st.markdown(f"- {point}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Results Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Review the generated outputs before downloading.</div>',
        unsafe_allow_html=True,
    )

    tab_names = [
        "Bulk Upload",
        "Bid Recommendations",
        "Search Term Actions",
        "Budget Actions",
        "SQP Opportunities",
        "AI Override Log",
    ]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)

    with tab1:
        if not combined_bulk_updates.empty:
            st.dataframe(combined_bulk_updates, use_container_width=True)
        else:
            st.info("No bulk upload output available.")

    with tab2:
        if not bid_recommendations.empty:
            st.dataframe(bid_recommendations, use_container_width=True)
        else:
            st.info("No bid recommendations available.")

    with tab3:
        if not search_term_actions.empty:
            st.dataframe(search_term_actions, use_container_width=True)
        else:
            st.info("No search term actions available.")

    with tab4:
        if not campaign_budget_actions.empty:
            st.dataframe(campaign_budget_actions, use_container_width=True)
        else:
            st.info("No campaign budget actions available.")

    with tab5:
        if not output_sqp_opportunities.empty:
            show_cols = [
                c for c in [
                    "search_query",
                    "search_query_score",
                    "search_query_volume",
                    "purchase_rate_pct",
                    "purchases_total_count",
                    "purchases_brand_share_pct",
                    "opportunity_tier",
                    "recommended_action",
                    "in_search_term_report",
                ] if c in output_sqp_opportunities.columns
            ]
            st.dataframe(output_sqp_opportunities[show_cols], use_container_width=True)
        else:
            st.info("No SQP opportunities available.")

    with tab6:
        if not ai_override_log_df.empty:
            st.dataframe(ai_override_log_df, use_container_width=True)
        else:
            st.info("No AI override log available for this run.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Downloads</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Export the files generated by this optimization run.</div>',
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2)
    d3, d4 = st.columns(2)

    with d1:
        st.download_button(
            label="Download Amazon Bulk Upload",
            data=to_excel_bytes(combined_bulk_updates),
            file_name="amazon_bulk_updates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_bulk_upload_xlsx",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            label="Download Bid Recommendations",
            data=to_excel_bytes(bid_recommendations),
            file_name="bid_recommendations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_bid_recommendations_xlsx",
            use_container_width=True,
        )

    with d3:
        st.download_button(
            label="Download Search Term Actions",
            data=to_excel_bytes(search_term_actions),
            file_name="search_term_actions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_search_term_actions_xlsx",
            use_container_width=True,
        )

    with d4:
        st.download_button(
            label="Download Campaign Budget Actions",
            data=to_excel_bytes(campaign_budget_actions),
            file_name="campaign_budget_actions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_campaign_budget_actions_xlsx",
            use_container_width=True,
        )

    if not output_sqp_opportunities.empty:
        st.download_button(
            label="Download SQP Opportunities",
            data=to_excel_bytes(output_sqp_opportunities),
            file_name="sqp_opportunities.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_sqp_opportunities_xlsx",
            use_container_width=True,
        )

    if not ai_override_log_df.empty:
        st.download_button(
            label="Download AI Override Log",
            data=to_excel_bytes(ai_override_log_df),
            file_name="ai_override_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_ai_override_log_xlsx",
            use_container_width=True,
        )

    with st.expander("Run History", expanded=False):
        if not run_history.empty:
            if "timestamp" in run_history.columns:
                st.dataframe(run_history.sort_values(by="timestamp", ascending=False), use_container_width=True)
            else:
                st.dataframe(run_history, use_container_width=True)
        else:
            st.info("No run history recorded yet.")


# =========================================================
# Footer
# =========================================================
footer_col1, footer_col2, footer_col3 = st.columns([3, 2, 3])
with footer_col2:
    st.markdown(
        '<div class="footer-note">Evolved Commerce · Amazon Ads Command Center</div>',
        unsafe_allow_html=True,
    )