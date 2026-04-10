import io
import os
import json
import calendar
from datetime import date
from typing import Any, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

from ads_optimizer_ingestion import Phase2UploadValidator, Phase2AdsOrchestrator


st.set_page_config(
    page_title="Evolved Commerce Amazon Ads Command Center",
    page_icon="📈",
    layout="wide",
)


# =========================================================
# Helpers
# =========================================================
BULK_EXPORT_COLUMNS = [
    "Product",
    "Entity",
    "Operation",
    "Campaign ID",
    "Ad Group ID",
    "Keyword ID",
    "Product Targeting ID",
    "Campaign Name",
    "Ad Group Name",
    "State",
    "Keyword Text",
    "Match Type",
    "Bid",
    "Budget",
    "Daily Budget",
    "Optimizer Action",
    "Confidence",
    "Score",
    "Reason",
]


RESULT_EXPORTS = {
    "execution_summary": "execution_summary.xlsx",
    "combined_bulk_updates": "combined_bulk_updates.xlsx",
    "bid_recommendations": "bid_recommendations.xlsx",
    "search_term_actions": "search_term_actions.xlsx",
    "campaign_budget_actions": "campaign_budget_actions.xlsx",
    "top_opportunities": "top_opportunities.xlsx",
    "sqp_opportunities": "sqp_opportunities.xlsx",
    "ai_override_log": "ai_override_log.xlsx",
    "placement_summary": "placement_summary.xlsx",
    "brand_segment_summary": "brand_segment_summary.xlsx",
    "trend_memory_summary": "trend_memory_summary.xlsx",
    "portfolio_reallocation_summary": "portfolio_reallocation_summary.xlsx",
}


def safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def get_number(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def get_uploaded_bytes(file_obj):
    return None if file_obj is None else file_obj.getvalue()


def bytes_to_buffer(file_bytes):
    return None if file_bytes is None else io.BytesIO(file_bytes)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def first_present_key(data: dict, keys: list[str], default=None):
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return default


def display_df(df: pd.DataFrame, preferred: Optional[list[str]] = None, height: int = 460) -> None:
    if df is None or df.empty:
        st.info("No data available.")
        return
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")].copy()
    if preferred:
        cols = [c for c in preferred if c in out.columns]
        extras = [c for c in out.columns if c not in cols][:10]
        out = out[cols + extras]
    st.dataframe(out, use_container_width=True, height=height)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    export_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    if export_df.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer, index=False, sheet_name="Output")
        return output.getvalue()

    export_df.columns = [str(c).strip() for c in export_df.columns]

    # First hard dedupe by header name
    export_df = export_df.loc[:, ~pd.Index(export_df.columns).duplicated(keep="first")].copy()

    approved_cols = []
    seen = set()
    for col in BULK_EXPORT_COLUMNS:
        if col in export_df.columns and col not in seen:
            approved_cols.append(col)
            seen.add(col)

    remaining_cols = []
    for col in export_df.columns:
        if col not in seen:
            remaining_cols.append(col)
            seen.add(col)

    export_df = export_df[approved_cols + remaining_cols].copy()

    # Final hard dedupe again after reordering
    export_df = export_df.loc[:, ~pd.Index(export_df.columns).duplicated(keep="first")].copy()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Output")
    return output.getvalue()


def product_bulk_slice(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    if df is None or df.empty or "Product" not in df.columns:
        return pd.DataFrame()

    out = df[df["Product"].astype(str).str.strip() == product_name].copy()
    out.columns = [str(c).strip() for c in out.columns]

    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")].copy()

    approved_cols = []
    seen = set()
    for col in BULK_EXPORT_COLUMNS:
        if col in out.columns and col not in seen:
            approved_cols.append(col)
            seen.add(col)

    remaining_cols = []
    for col in out.columns:
        if col not in seen:
            remaining_cols.append(col)
            seen.add(col)

    out = out[approved_cols + remaining_cols].copy()
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")].copy()

    return out.reset_index(drop=True)


def load_logo_path() -> Optional[str]:
    for path in [
        "assets/ec_logo.png",
        "assets/ec_logo.jpg",
        "assets/ec_logo.jpeg",
        "assets/logo.png",
        "assets/logo.jpg",
        "ec_logo.png",
        "logo.png",
    ]:
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


def render_info_banner(title: str, body: str, tone: str = "brand") -> None:
    border = {"brand": "#F97316", "good": "#10B981", "warn": "#F59E0B", "bad": "#EF4444"}.get(tone, "#F97316")
    bg = {"brand": "#FFF7ED", "good": "#ECFDF5", "warn": "#FFFBEB", "bad": "#FEF2F2"}.get(tone, "#FFF7ED")
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border:1px solid {border};
            border-left:6px solid {border};
            border-radius:14px;
            padding:14px 16px;
            margin: 0.4rem 0 1rem 0;
        ">
            <div style="font-weight:800; color:#111827; margin-bottom:4px;">{title}</div>
            <div style="color:#374151; font-size:0.94rem;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_readiness_block(label: str, ready_state: dict, note: str = "") -> None:
    status = str(ready_state.get("status", "Missing"))
    tone = "good" if ready_state.get("ready") else ("warn" if status == "Partial" else "bad")
    missing = safe_list(ready_state.get("missing_required"))
    render_metric_card(label, status, tone=tone, small=True)
    if missing:
        st.caption("Missing: " + ", ".join(missing))
    elif note:
        st.caption(note)


def get_scaling_action_count(df: pd.DataFrame) -> int:
    if df is None or df.empty or "Optimizer Action" not in df.columns:
        return 0
    return int(df["Optimizer Action"].astype(str).str.upper().isin(["INCREASE_BID", "INCREASE_BUDGET"]).sum())


def estimate_tacos_suppressed_scaling(combined_bulk_updates: pd.DataFrame, account_health: dict, simulation_summary: dict) -> dict:
    if combined_bulk_updates is None or combined_bulk_updates.empty:
        return {"suppressed_bid_increases": 0, "suppressed_budget_increases": 0, "suppressed_total": 0}
    if str(account_health.get("tacos_status", "not_used")).lower() != "above_target":
        return {"suppressed_bid_increases": 0, "suppressed_budget_increases": 0, "suppressed_total": 0}
    bid_sup = int(simulation_summary.get("bid_increases", 0) or 0)
    budget_sup = int(simulation_summary.get("budget_increases", 0) or 0)
    return {
        "suppressed_bid_increases": bid_sup,
        "suppressed_budget_increases": budget_sup,
        "suppressed_total": bid_sup + budget_sup,
    }


def build_conflicting_signals(account_health: dict, simulation_summary: dict, account_summary: dict) -> list[str]:
    messages = []
    tacos_status = str(account_health.get("tacos_status", "not_used")).lower()
    health_status = str(account_health.get("health_status", "unknown")).lower()
    scalable = get_int(account_summary.get("campaigns_scalable"))
    bid_increases = get_int(simulation_summary.get("bid_increases"))
    budget_increases = get_int(simulation_summary.get("budget_increases"))

    if tacos_status == "above_target" and scalable > 0:
        messages.append("Scaling opportunities exist, but account-wide TACOS is above target, so growth is being constrained.")
    if health_status == "under_target" and (bid_increases > 0 or budget_increases > 0):
        messages.append("Some entities still qualify for scaling, but overall account efficiency is below target, so scaling should be reviewed carefully.")
    if scalable > 0 and bid_increases == 0 and budget_increases == 0:
        messages.append("The account has scalable campaigns, but no scaling actions were generated. A guardrail is likely suppressing growth.")
    return messages


def build_narrative(account_health: dict, simulation_summary: dict, pacing_status: str) -> list[str]:
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
        notes.append(f"Account ROAS is healthy at {account_roas}. The optimizer allowed more room for controlled scaling.")
    elif health_status == "tacos_constrained":
        notes.append("TACOS guardrails are active. The optimizer tightened scaling behavior to protect total account efficiency.")
    else:
        notes.append(f"Account health is stable. The optimizer used a balanced rule set around the target ROAS of {adjusted_min_roas}.")

    if pacing_status == "Over Pace":
        notes.append("Monthly pacing is currently over target. Budget increases and bid increases were suppressed to help protect the monthly spend cap.")
    elif pacing_status == "On Pace":
        notes.append("Monthly pacing is on target. Standard optimization behavior was allowed without pacing suppression.")

    if simulation_summary.get("bid_decreases", 0) > 0:
        notes.append(f"{simulation_summary['bid_decreases']} bid decreases were generated to reduce waste and tighten weak targets.")
    if simulation_summary.get("bid_increases", 0) > 0:
        notes.append(f"{simulation_summary['bid_increases']} bid increases were generated for strong-performing targets with scaling headroom.")
    if simulation_summary.get("harvested_keywords", 0) > 0:
        notes.append(f"{simulation_summary['harvested_keywords']} search terms were harvested into Exact keywords based on conversion and efficiency thresholds.")
    if simulation_summary.get("negatives_added", 0) > 0:
        notes.append(f"{simulation_summary['negatives_added']} negative phrases were added to reduce wasted clicks from unproductive traffic.")
    if simulation_summary.get("budget_increases", 0) > 0 or simulation_summary.get("budget_decreases", 0) > 0:
        notes.append(
            f"Campaign budget actions included {simulation_summary.get('budget_increases', 0)} increases and {simulation_summary.get('budget_decreases', 0)} decreases."
        )
    if simulation_summary.get("high_confidence_actions", 0) > 0:
        notes.append(f"{simulation_summary.get('high_confidence_actions', 0)} actions were tagged high confidence by the scoring engine.")
    if tacos_pct is not None:
        notes.append(f"Current TACOS reading from this run is {tacos_pct}%.")
    return notes


def summarize_placement_actions(bid_recommendations: pd.DataFrame) -> pd.DataFrame:
    if bid_recommendations is None or bid_recommendations.empty or "placement_bucket" not in bid_recommendations.columns:
        return pd.DataFrame()
    df = bid_recommendations.copy()
    if "recommended_action" not in df.columns:
        return pd.DataFrame()
    summary = (
        df.groupby("placement_bucket", dropna=False)
        .agg(
            rows=("placement_bucket", "size"),
            increase_bid=("recommended_action", lambda s: int((s.astype(str) == "INCREASE_BID").sum())),
            decrease_bid=("recommended_action", lambda s: int((s.astype(str) == "DECREASE_BID").sum())),
            avg_roas=("roas", "mean"),
            avg_clicks=("clicks", "mean"),
            avg_orders=("orders", "mean"),
            avg_score=("score", "mean") if "score" in df.columns else ("placement_bucket", "size"),
        )
        .reset_index()
    )
    for col in ["avg_roas", "avg_clicks", "avg_orders", "avg_score"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").round(2)
    return summary.sort_values(["increase_bid", "rows"], ascending=[False, False]).reset_index(drop=True)


def summarize_brand_segments(bid_recommendations: pd.DataFrame, search_term_actions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if isinstance(bid_recommendations, pd.DataFrame) and not bid_recommendations.empty and "brand_segment" in bid_recommendations.columns:
        b = bid_recommendations.copy()
        b["action_type"] = b.get("recommended_action", "")
        frames.append(b[[c for c in ["brand_segment", "action_type", "roas", "clicks", "orders", "score"] if c in b.columns]])
    if isinstance(search_term_actions, pd.DataFrame) and not search_term_actions.empty and "brand_segment" in search_term_actions.columns:
        s = search_term_actions.copy()
        s["action_type"] = s.get("search_term_action", "")
        frames.append(s[[c for c in ["brand_segment", "action_type", "roas", "clicks", "orders", "score"] if c in s.columns]])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    summary = (
        df.groupby("brand_segment", dropna=False)
        .agg(
            rows=("brand_segment", "size"),
            increase_actions=("action_type", lambda s: int(s.astype(str).str.contains("INCREASE|HARVEST", case=False, regex=True).sum())),
            decrease_actions=("action_type", lambda s: int(s.astype(str).str.contains("DECREASE|NEGATIVE", case=False, regex=True).sum())),
            avg_roas=("roas", "mean") if "roas" in df.columns else ("brand_segment", "size"),
            avg_clicks=("clicks", "mean") if "clicks" in df.columns else ("brand_segment", "size"),
            avg_orders=("orders", "mean") if "orders" in df.columns else ("brand_segment", "size"),
        )
        .reset_index()
    )
    for col in ["avg_roas", "avg_clicks", "avg_orders"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").round(2)
    return summary.sort_values(["rows"], ascending=[False]).reset_index(drop=True)


def summarize_trend_memory(bid_recommendations: pd.DataFrame, campaign_budget_actions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for df, level, action_col in [
        (bid_recommendations, "target", "recommended_action"),
        (campaign_budget_actions, "campaign", "campaign_action"),
    ]:
        if isinstance(df, pd.DataFrame) and not df.empty and "cooldown_active" in df.columns:
            keep = [c for c in ["campaign_name", "ad_group_name", "target", "roas_trend", "click_trend", "order_trend", "prior_action_direction", "cooldown_active", action_col] if c in df.columns]
            x = df[keep].copy()
            x["entity_level"] = level
            x["final_action"] = df[action_col] if action_col in df.columns else ""
            frames.append(x)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    summary = (
        df.groupby(["entity_level", "cooldown_active", "roas_trend", "prior_action_direction"], dropna=False)
        .agg(rows=("entity_level", "size"))
        .reset_index()
        .sort_values(["rows"], ascending=[False])
        .reset_index(drop=True)
    )
    return summary


def summarize_portfolio_reallocation(campaign_budget_actions: pd.DataFrame) -> pd.DataFrame:
    if campaign_budget_actions is None or campaign_budget_actions.empty:
        return pd.DataFrame()
    needed_cols = {"portfolio_key", "reallocation_role", "reallocation_delta"}
    if not needed_cols.issubset(set(campaign_budget_actions.columns)):
        return pd.DataFrame()
    df = campaign_budget_actions.copy()
    summary = (
        df.groupby(["portfolio_key", "portfolio_name", "reallocation_role"], dropna=False)
        .agg(
            rows=("portfolio_key", "size"),
            total_delta=("reallocation_delta", "sum"),
            avg_score=("score", "mean") if "score" in df.columns else ("portfolio_key", "size"),
        )
        .reset_index()
    )
    if "avg_score" in summary.columns:
        summary["avg_score"] = pd.to_numeric(summary["avg_score"], errors="coerce").round(3)
    summary["total_delta"] = pd.to_numeric(summary["total_delta"], errors="coerce").round(2)
    return summary.sort_values(["portfolio_key", "reallocation_role"]).reset_index(drop=True)


# =========================================================
# AI Layer
# =========================================================
def get_openai_api_key() -> Optional[str]:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_openai_client() -> Optional[OpenAI]:
    api_key = get_openai_api_key()
    return OpenAI(api_key=api_key) if api_key else None


def get_openai_model_candidates() -> list[str]:
    preferred = []
    try:
        if "OPENAI_MODEL" in st.secrets and st.secrets["OPENAI_MODEL"]:
            preferred.append(str(st.secrets["OPENAI_MODEL"]).strip())
    except Exception:
        pass
    env_model = os.getenv("OPENAI_MODEL", "").strip()
    if env_model:
        preferred.append(env_model)
    fallbacks = ["gpt-5", "gpt-5-mini", "gpt-4o"]
    out, seen = [], set()
    for model in preferred + fallbacks:
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out


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


@st.cache_data(show_spinner=False, ttl=900)
def run_ai_review_cached(payload_json: str) -> dict:
    client = get_openai_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY not found.")

    instructions = """
You are an expert Amazon Ads optimizer.

You are reviewing ONLY low-confidence optimization actions. Do not review high-confidence actions.

Your goals:
- KEEP actions that still look directionally correct.
- MODIFY actions only when there is a clearly safer alternative.
- REMOVE actions that look risky or unsupported.
- Use placement, branded/non-branded context, trend signals, and portfolio reallocation context when present.

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

    schema = {
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
                        "decision": {"type": "string", "enum": ["KEEP", "MODIFY", "REMOVE"]},
                        "new_action": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "decision", "new_action", "reason"],
                },
            },
        },
        "required": ["executive_summary", "overrides"],
    }

    last_error = None
    for model in get_openai_model_candidates():
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=payload_json,
                max_output_tokens=1400,
                text={"format": {"type": "json_schema", "name": "ai_overrides", "schema": schema}},
            )
            raw_text = ""
            if hasattr(response, "output_text") and response.output_text:
                raw_text = str(response.output_text).strip()
            if not raw_text and hasattr(response, "output") and response.output:
                parts = []
                for item in response.output:
                    contents = getattr(item, "content", None)
                    if not contents:
                        continue
                    for content_item in contents:
                        text_value = getattr(content_item, "text", None)
                        if text_value:
                            parts.append(str(getattr(text_value, "value", text_value)))
                        value_value = getattr(content_item, "value", None)
                        if value_value:
                            parts.append(str(value_value))
                raw_text = "\n".join([p for p in parts if str(p).strip()]).strip()
            if not raw_text:
                raise ValueError(f"{model}: OpenAI returned no readable text content.")
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(raw_text[start:end + 1])
                else:
                    raise ValueError(f"{model}: OpenAI returned non-JSON content.")
            parsed["_model_used"] = model
            return parsed
        except Exception as e:
            last_error = e
            continue
    raise ValueError(f"All AI model fallbacks failed. Last error: {last_error}")


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
        bid["_campaign_key"] = bid.get("campaign_name", pd.Series(dtype=str)).map(normalize_text)
        bid["_ad_group_key"] = bid.get("ad_group_name", pd.Series(dtype=str)).map(normalize_text)
        bid["_term_key"] = bid.get("target", pd.Series(dtype=str)).map(normalize_text)
        bid["_match_key"] = bid.get("match_type", pd.Series(dtype=str)).map(normalize_text)

    if not search.empty:
        search["_campaign_key"] = search.get("campaign_name", pd.Series(dtype=str)).map(normalize_text)
        search["_ad_group_key"] = search.get("ad_group_name", pd.Series(dtype=str)).map(normalize_text)
        search["_term_key"] = search.get("search_term", pd.Series(dtype=str)).map(normalize_text)
        search["_action_key"] = search.get("search_term_action", pd.Series(dtype=str)).astype(str).str.upper()

    if not budget.empty:
        budget["_campaign_key"] = budget.get("campaign_name", pd.Series(dtype=str)).map(normalize_text)
        budget["_action_key"] = budget.get("campaign_action", pd.Series(dtype=str)).astype(str).str.upper()

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
            "score": get_number(row.get("Score")),
            "reason": str(row.get("Reason", "")),
            "confidence": str(row.get("Confidence", "")).upper().strip(),
            "impression_share_pct": 0.0,
            "current_bid": None,
            "recommended_bid": None,
            "current_daily_budget": None,
            "recommended_daily_budget": None,
            "brand_segment": "",
            "placement_bucket": "",
            "roas_trend": "",
            "click_trend": "",
            "order_trend": "",
            "prior_action_direction": "",
            "portfolio_name": "",
            "reallocation_role": "",
        }

        campaign_key = normalize_text(row.get("Campaign Name", ""))
        ad_group_key = normalize_text(row.get("Ad Group Name", ""))
        term_key = normalize_text(row.get("Keyword Text", ""))
        match_key = normalize_text(row.get("Match Type", ""))
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
                candidate.update({
                    "source_type": "bid",
                    "clicks": get_number(m.get("clicks")),
                    "orders": get_number(m.get("orders")),
                    "spend": get_number(m.get("spend")),
                    "sales": get_number(m.get("sales")),
                    "roas": get_number(m.get("roas")),
                    "impression_share_pct": get_number(m.get("impression_share_pct")),
                    "current_bid": get_number(m.get("current_bid")),
                    "recommended_bid": get_number(m.get("recommended_bid")),
                    "score": get_number(first_present_key(m, ["score", "Score"])),
                    "reason": str(first_present_key(m, ["reason", "Reason"], "") or ""),
                    "confidence": str(first_present_key(m, ["confidence", "Confidence"], "") or "").upper(),
                    "brand_segment": str(m.get("brand_segment", "") or ""),
                    "placement_bucket": str(m.get("placement_bucket", "") or ""),
                    "roas_trend": str(m.get("roas_trend", "") or ""),
                    "click_trend": str(m.get("click_trend", "") or ""),
                    "order_trend": str(m.get("order_trend", "") or ""),
                    "prior_action_direction": str(m.get("prior_action_direction", "") or ""),
                })
        elif optimizer_action in {"HARVEST_TO_EXACT", "ADD_NEGATIVE_PHRASE"} and not search.empty:
            match = search[
                (search["_campaign_key"] == campaign_key)
                & (search["_ad_group_key"] == ad_group_key)
                & (search["_term_key"] == term_key)
                & (search["_action_key"] == optimizer_action)
            ]
            if not match.empty:
                m = match.iloc[0]
                candidate.update({
                    "source_type": "search_harvest" if optimizer_action == "HARVEST_TO_EXACT" else "negative_keyword",
                    "clicks": get_number(m.get("clicks")),
                    "orders": get_number(m.get("orders")),
                    "spend": get_number(m.get("spend")),
                    "sales": get_number(m.get("sales")),
                    "roas": get_number(m.get("roas")),
                    "score": get_number(first_present_key(m, ["score", "Score"])),
                    "reason": str(first_present_key(m, ["reason", "Reason"], "") or ""),
                    "confidence": str(first_present_key(m, ["confidence", "Confidence"], "") or "").upper(),
                    "brand_segment": str(m.get("brand_segment", "") or ""),
                    "placement_bucket": str(m.get("placement_bucket", "") or ""),
                    "roas_trend": str(m.get("roas_trend", "") or ""),
                    "click_trend": str(m.get("click_trend", "") or ""),
                    "order_trend": str(m.get("order_trend", "") or ""),
                    "prior_action_direction": str(m.get("prior_action_direction", "") or ""),
                })
        elif entity == "Campaign" and optimizer_action in {"INCREASE_BUDGET", "DECREASE_BUDGET"} and not budget.empty:
            match = budget[(budget["_campaign_key"] == campaign_key) & (budget["_action_key"] == optimizer_action)]
            if not match.empty:
                m = match.iloc[0]
                candidate.update({
                    "source_type": "budget",
                    "clicks": get_number(m.get("clicks")),
                    "orders": get_number(m.get("orders")),
                    "spend": get_number(m.get("spend")),
                    "sales": get_number(m.get("sales")),
                    "roas": get_number(m.get("roas")),
                    "impression_share_pct": get_number(m.get("avg_impression_share_pct")),
                    "current_daily_budget": get_number(m.get("daily_budget")),
                    "recommended_daily_budget": get_number(m.get("recommended_daily_budget")),
                    "score": get_number(first_present_key(m, ["score", "Score"])),
                    "reason": str(first_present_key(m, ["reason", "Reason"], "") or ""),
                    "confidence": str(first_present_key(m, ["confidence", "Confidence"], "") or "").upper(),
                    "brand_segment": str(m.get("brand_segment", "") or ""),
                    "roas_trend": str(m.get("roas_trend", "") or ""),
                    "click_trend": str(m.get("click_trend", "") or ""),
                    "order_trend": str(m.get("order_trend", "") or ""),
                    "prior_action_direction": str(m.get("prior_action_direction", "") or ""),
                    "portfolio_name": str(m.get("portfolio_name", "") or ""),
                    "reallocation_role": str(m.get("reallocation_role", "") or ""),
                })

        candidate["confidence"] = candidate.get("confidence") or score_action_confidence(candidate)
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
        "id", "source_type", "optimizer_action", "campaign_name", "ad_group_name", "keyword_text", "match_type",
        "clicks", "orders", "spend", "sales", "roas", "impression_share_pct", "current_bid", "recommended_bid",
        "current_daily_budget", "recommended_daily_budget", "score", "reason", "confidence", "brand_segment",
        "placement_bucket", "roas_trend", "click_trend", "order_trend", "prior_action_direction", "portfolio_name",
        "reallocation_role",
    ]
    cols = [c for c in keep_cols if c in low_conf_df.columns]

    sqp_context = []
    if sqp_opportunities is not None and not sqp_opportunities.empty:
        keep_sqp_cols = [
            c for c in [
                "search_query", "search_query_score", "search_query_volume", "purchase_rate_pct", "purchases_total_count",
                "purchases_brand_share_pct", "opportunity_tier", "recommended_action", "in_search_term_report",
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


def get_strategy_parameters(strategy_mode: str) -> dict:
    mode = str(strategy_mode).strip().lower()
    if mode == "conservative":
        return {"max_bid_up": 0.05, "max_bid_down": 0.10, "budget_up_pct": 0.05, "budget_down_pct": 0.08}
    if mode == "aggressive":
        return {"max_bid_up": 0.20, "max_bid_down": 0.25, "budget_up_pct": 0.15, "budget_down_pct": 0.12}
    return {"max_bid_up": 0.10, "max_bid_down": 0.15, "budget_up_pct": 0.10, "budget_down_pct": 0.10}


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
    candidate_map = {str(row["id"]): row.to_dict() for _, row in ai_candidates_df.iterrows() if "id" in ai_candidates_df.columns}

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
                        row_dict["Bid"] = round_bid_value(current_bid * (1 + strategy["max_bid_up"]), max_bid_cap)
                    else:
                        row_dict["Bid"] = round_bid_value(current_bid * (1 - strategy["max_bid_down"]), max_bid_cap)
                elif source_type == "budget" and new_action in {"INCREASE_BUDGET", "DECREASE_BUDGET"} and current_budget > 0:
                    final_action = new_action
                    if new_action == "INCREASE_BUDGET":
                        row_dict["Daily Budget"] = round_budget_value(current_budget * (1 + strategy["budget_up_pct"]), max_budget_cap)
                    else:
                        row_dict["Daily Budget"] = round_budget_value(current_budget * (1 - strategy["budget_down_pct"]), max_budget_cap)
                elif source_type == "search_harvest" and new_action == "HARVEST_TO_EXACT":
                    final_action = new_action
                elif source_type == "negative_keyword" and new_action == "ADD_NEGATIVE_PHRASE":
                    final_action = new_action

        if not was_removed:
            row_dict["Optimizer Action"] = final_action
            updated_rows.append(row_dict)

        candidate = candidate_map.get(rid, {})
        log_rows.append({
            "ID": rid,
            "Source Type": source_type,
            "Campaign Name": candidate.get("campaign_name", row_dict.get("Campaign Name", "")),
            "Ad Group Name": candidate.get("ad_group_name", row_dict.get("Ad Group Name", "")),
            "Keyword Text": candidate.get("keyword_text", row_dict.get("Keyword Text", "")),
            "Confidence": candidate.get("confidence", ""),
            "Score": candidate.get("score", 0),
            "Clicks": candidate.get("clicks", 0),
            "Orders": candidate.get("orders", 0),
            "Spend": candidate.get("spend", 0),
            "Sales": candidate.get("sales", 0),
            "ROAS": candidate.get("roas", 0),
            "Brand Segment": candidate.get("brand_segment", ""),
            "Placement": candidate.get("placement_bucket", ""),
            "ROAS Trend": candidate.get("roas_trend", ""),
            "Original Action": original_action,
            "Decision": decision,
            "Final Action": "REMOVED" if was_removed else final_action,
            "Reason": reason,
        })

    return pd.DataFrame(updated_rows), pd.DataFrame(log_rows)


def build_ai_impact_summary(ai_override_log_df: pd.DataFrame, original_count: int, final_count: int, executive_summary: str) -> dict:
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
    removed_rows = ai_override_log_df[ai_override_log_df["Final Action"] == "REMOVED"]
    return {
        "reviewed": len(ai_override_log_df),
        "kept": int((ai_override_log_df["Decision"] == "KEEP").sum()),
        "modified": int((ai_override_log_df["Decision"] == "MODIFY").sum()),
        "removed": int((ai_override_log_df["Final Action"] == "REMOVED").sum()),
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
        .main > div { padding-top: 1rem; }
        .block-container { padding-top: 1rem; padding-bottom: 1.5rem; max-width: 1440px; }
        [data-testid="stSidebar"] { background: #F3F4F6; }
        .brand-shell { background: linear-gradient(135deg, #EA580C 0%, #1F2937 100%); border-radius: 20px; padding: 58px 30px; color: white; margin-bottom: 1.1rem; box-shadow: 0 12px 28px rgba(0,0,0,.12); }
        .brand-title { font-size: 2.1rem; font-weight: 800; line-height: 1.05; margin: 0; }
        .brand-subtitle { font-size: 1rem; opacity: .96; margin-top: .55rem; }
        .section-title { font-size: 1.18rem; font-weight: 760; color: #111827; margin-bottom: .18rem; }
        .section-note { color: #4B5563; font-size: .94rem; margin-bottom: .9rem; }
        .section-divider { border-top: 1px solid #E5E7EB; margin: 1rem 0 1.1rem 0; }
        .metric-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px; padding: 12px 14px; box-shadow: 0 4px 14px rgba(15,23,42,.05); min-height: 80px; }
        .metric-label { font-size: .75rem; color: #6B7280; margin-bottom: .25rem; font-weight: 600; }
        .metric-value { font-size: 1.5rem; line-height: 1.1; font-weight: 800; color: #1F2937; word-break: break-word; }
        .metric-value.small { font-size: 1.15rem; }
        .metric-card.good { border-left: 4px solid #10B981; }
        .metric-card.warn { border-left: 4px solid #F59E0B; }
        .metric-card.bad { border-left: 4px solid #EF4444; }
        .metric-card.brand { border-left: 4px solid #F47322; }
        .footer-note { color: #6B7280; font-size: .85rem; text-align: center; opacity: .9; margin-top: .35rem; margin-bottom: .5rem; }
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
- Supports explainability, confidence scoring, simulation reporting, and Amazon-only business-aware optimization
- Surfaces placement-aware, branded/non-branded, trend-memory, and portfolio-reallocation outputs when returned by the ingestion layer
"""
    )
    st.markdown("---")
    st.markdown("## Required uploads")
    st.markdown(
        """
Shared
- Bulk Sheet

Sponsored Products
- Search Term Report
- Targeting Report
- Impression Share Report

Sponsored Brands
- Search Term Report

Sponsored Display
- Targeting Report
"""
    )
    st.markdown("## Optional uploads")
    st.markdown(
        """
- Sales and Traffic Business Report for TACOS control and Amazon-only business-aware optimization
- Search Query Performance Report (Simple View, prior month only)
"""
    )
    st.markdown("---")
    st.markdown("## AI Optimization Layer")
    enable_ai_review = st.checkbox("Enable AI Optimization Layer", value=True)
    api_key_present = bool(get_openai_api_key())
    if enable_ai_review:
        st.caption("AI reviews only low-confidence actions and can refine the final output.")
        st.caption(f"Preferred AI Model: {get_openai_model_candidates()[0]}")
    if enable_ai_review and not api_key_present:
        st.warning("OPENAI_API_KEY not found. Add it to Streamlit secrets or environment variables.")
    st.markdown("---")
    st.markdown("**Version:** 2.1")
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
                Bid optimization • Search harvesting • Negative mining • Budget pacing • TACOS control • SQP context • AI review • Explainability • Amazon-only business context
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_info_banner(
    "Upgraded app layer",
    "This version is cleaned up for the upgraded ingestion engine and adds reporting for placement optimization, branded segmentation, trend memory, and portfolio budget reallocation outputs.",
    tone="brand",
)


# =========================================================
# Settings
# =========================================================
st.markdown('<div class="section-title">Optimization Settings</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Core controls for bidding, efficiency, and zero-order action logic.</div>', unsafe_allow_html=True)

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    min_roas = st.number_input("Minimum ROAS Target", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
with r1c2:
    min_clicks = st.number_input("Minimum Clicks Before Standard Efficiency Actions", min_value=1, max_value=100, value=8, step=1)
with r1c3:
    strategy_mode = st.selectbox("Strategy Mode", options=["Conservative", "Balanced", "Aggressive"], index=1)

r2c1, r2c2 = st.columns(2)
with r2c1:
    losing_kw_click_threshold = st.number_input("Losing KW Click Threshold", min_value=1, max_value=100, value=12, step=1)
with r2c2:
    losing_kw_action = st.selectbox("Losing KW Action Type", options=["Decrease Bid", "Add Negative", "Both", "None"], index=2)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Optimization Actions</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Enable or disable the main optimization behaviors for this run.</div>', unsafe_allow_html=True)

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
    eg1, eg2 = st.columns(2)
    with eg1:
        enable_tacos_control = st.checkbox("Enable TACOS Control", value=False)
    with eg2:
        max_tacos_target = st.number_input("Maximum TACOS %", min_value=1.0, max_value=100.0, value=15.0, step=0.5, disabled=not enable_tacos_control)

with st.expander("Budget Guardrails", expanded=False):
    bg1, bg2 = st.columns(2)
    with bg1:
        enable_monthly_budget_control = st.checkbox("Enable Monthly Budget Control", value=False)
    with bg2:
        pacing_buffer_pct = st.number_input("Pacing Buffer %", min_value=0.0, max_value=50.0, value=5.0, step=1.0, disabled=not enable_monthly_budget_control)
    bg3, bg4 = st.columns(2)
    with bg3:
        monthly_account_budget = st.number_input("Monthly Account Budget", min_value=0.0, value=0.0, step=100.0, disabled=not enable_monthly_budget_control)
    with bg4:
        month_to_date_spend = st.number_input("Month-to-Date Ad Spend", min_value=0.0, value=0.0, step=10.0, disabled=not enable_monthly_budget_control)
    bg5, bg6 = st.columns(2)
    with bg5:
        max_bid_cap = st.number_input("Maximum Recommended Bid", min_value=0.05, max_value=20.0, value=5.0, step=0.05)
    with bg6:
        max_budget_cap = st.number_input("Maximum Recommended Daily Budget", min_value=1.0, max_value=10000.0, value=500.0, step=1.0)

render_info_banner(
    "Advanced logic in this build",
    "Placement-aware weighting, branded/non-branded segmentation, trend-memory anti-oscillation logic, and portfolio budget reallocation are handled inside the ingestion engine and reported below when available.",
    tone="good",
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
st.markdown('<div class="section-note">Provide shared reports plus SP / SB / SD report files. This version uses only Amazon Ads Manager and Seller Central data sources.</div>', unsafe_allow_html=True)

with st.expander("Shared Uploads", expanded=True):
    su1, su2 = st.columns(2)
    with su1:
        bulk_file = st.file_uploader("Bulk Sheet", type=["xlsx"], key="bulk")
        upload_status_line(bulk_file, "Bulk Sheet uploaded")
    with su2:
        business_file = st.file_uploader("Sales and Traffic Business Report (optional — used for TACOS and Amazon-only business-aware optimization)", type=["xlsx", "csv"], key="business")
        upload_status_line(business_file, "Business report uploaded")
    sqp_file = st.file_uploader("Search Query Performance Report (optional — prior month, Simple View only)", type=["csv"], key="sqp", help="Use the prior month's SQP report in Simple View only.")
    upload_status_line(sqp_file, "SQP report uploaded")

with st.expander("Sponsored Products Uploads", expanded=True):
    sp1, sp2, sp3 = st.columns(3)
    with sp1:
        sp_search_file = st.file_uploader("SP Search Term Report", type=["xlsx"], key="sp_search")
        upload_status_line(sp_search_file, "SP Search Term Report uploaded")
    with sp2:
        sp_targeting_file = st.file_uploader("SP Targeting Report", type=["xlsx"], key="sp_targeting")
        upload_status_line(sp_targeting_file, "SP Targeting Report uploaded")
    with sp3:
        sp_impression_file = st.file_uploader("SP Impression Share Report", type=["csv"], key="sp_impression")
        upload_status_line(sp_impression_file, "SP Impression Share Report uploaded")

with st.expander("Sponsored Brands Uploads", expanded=False):
    sb1, sb2 = st.columns(2)
    with sb1:
        sb_search_file = st.file_uploader("SB Search Term Report", type=["xlsx"], key="sb_search")
        upload_status_line(sb_search_file, "SB Search Term Report uploaded")
    with sb2:
        sb_impression_file = st.file_uploader("SB Impression Share Report (optional)", type=["csv", "xlsx"], key="sb_impression")
        upload_status_line(sb_impression_file, "SB Impression Share Report uploaded")

with st.expander("Sponsored Display Uploads", expanded=False):
    sd_targeting_file = st.file_uploader("SD Targeting Report", type=["xlsx"], key="sd_targeting")
    upload_status_line(sd_targeting_file, "SD Targeting Report uploaded")

bulk_bytes = get_uploaded_bytes(bulk_file)
business_bytes = get_uploaded_bytes(business_file)
sqp_bytes = get_uploaded_bytes(sqp_file)
sp_search_bytes = get_uploaded_bytes(sp_search_file)
sp_targeting_bytes = get_uploaded_bytes(sp_targeting_file)
sp_impression_bytes = get_uploaded_bytes(sp_impression_file)
sb_search_bytes = get_uploaded_bytes(sb_search_file)
sb_impression_bytes = get_uploaded_bytes(sb_impression_file)
sd_targeting_bytes = get_uploaded_bytes(sd_targeting_file)


def build_engine() -> Phase2AdsOrchestrator:
    return Phase2AdsOrchestrator(
        bulk_file=bytes_to_buffer(bulk_bytes),
        business_report_file=bytes_to_buffer(business_bytes),
        sqp_report_file=bytes_to_buffer(sqp_bytes),
        sp_search_term_file=bytes_to_buffer(sp_search_bytes),
        sp_targeting_file=bytes_to_buffer(sp_targeting_bytes),
        sp_impression_share_file=bytes_to_buffer(sp_impression_bytes),
        sb_search_term_file=bytes_to_buffer(sb_search_bytes),
        sb_impression_share_file=bytes_to_buffer(sb_impression_bytes),
        sd_targeting_file=bytes_to_buffer(sd_targeting_bytes),
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
# Validation + Diagnostics
# =========================================================
validation = {}
readiness = {}
spend_summary = {}
runnable_types = []
bulk_sheet_names = []
diagnostics = {}
account_health = {}
account_summary = {}
preview = {}
smart_warnings = []
optimization_suggestions = []
campaign_health_dashboard = pd.DataFrame()
sqp_opportunities = pd.DataFrame()
sqp_summary = {}
top_opportunities = pd.DataFrame()

if bulk_bytes is not None:
    try:
        validator = Phase2UploadValidator(
            bulk_file=bytes_to_buffer(bulk_bytes),
            business_report_file=bytes_to_buffer(business_bytes),
            sqp_report_file=bytes_to_buffer(sqp_bytes),
            sp_search_term_file=bytes_to_buffer(sp_search_bytes),
            sp_targeting_file=bytes_to_buffer(sp_targeting_bytes),
            sp_impression_share_file=bytes_to_buffer(sp_impression_bytes),
            sb_search_term_file=bytes_to_buffer(sb_search_bytes),
            sb_impression_share_file=bytes_to_buffer(sb_impression_bytes),
            sd_targeting_file=bytes_to_buffer(sd_targeting_bytes),
        )
        validation = safe_dict(validator.analyze())
        readiness = safe_dict(validation.get("readiness"))
        spend_summary = safe_dict(validation.get("spend_summary"))
        runnable_types = safe_list(validation.get("runnable_types"))
        bulk_sheet_names = safe_list(validation.get("bulk_sheet_names"))
    except Exception as e:
        st.error(f"Upload validation failed: {e}")

sp_ready = safe_dict(readiness.get("SP")).get("ready", False)
tacos_ready = (not enable_tacos_control) or (business_bytes is not None)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Optimizer Readiness Validation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Validates upload readiness across Sponsored Products, Sponsored Brands, and Sponsored Display before any optimization runs.</div>', unsafe_allow_html=True)

vr1, vr2, vr3 = st.columns(3)
with vr1:
    render_readiness_block("Sponsored Products", safe_dict(readiness.get("SP")), note="SP drives the deepest diagnostics.")
with vr2:
    render_readiness_block("Sponsored Brands", safe_dict(readiness.get("SB")))
with vr3:
    render_readiness_block("Sponsored Display", safe_dict(readiness.get("SD")))

if bulk_sheet_names:
    st.caption("Bulk tabs found: " + ", ".join(bulk_sheet_names))

if readiness:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Spend Reconciliation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">This separates SP, SB, and SD spend so you can compare the optimizer view to total account spend.</div>', unsafe_allow_html=True)
    sr1, sr2, sr3, sr4 = st.columns(4)
    with sr1:
        render_metric_card("SP Spend", f"${get_number(spend_summary.get('sp_spend')):,.2f}", tone="brand")
    with sr2:
        render_metric_card("SB Spend", f"${get_number(spend_summary.get('sb_spend')):,.2f}", tone="brand")
    with sr3:
        render_metric_card("SD Spend", f"${get_number(spend_summary.get('sd_spend')):,.2f}", tone="brand")
    with sr4:
        render_metric_card("Total Spend", f"${get_number(spend_summary.get('total_spend')):,.2f}", tone="good")
    ss1, ss2, ss3, ss4 = st.columns(4)
    with ss1:
        render_metric_card("SP Sales", f"${get_number(spend_summary.get('sp_sales')):,.2f}", tone="brand")
    with ss2:
        render_metric_card("SB Sales", f"${get_number(spend_summary.get('sb_sales')):,.2f}", tone="brand")
    with ss3:
        render_metric_card("SD Sales", f"${get_number(spend_summary.get('sd_sales')):,.2f}", tone="brand")
    with ss4:
        render_metric_card("Total Sales", f"${get_number(spend_summary.get('total_sales')):,.2f}", tone="good")
    if runnable_types:
        st.success("Runnable ad types from the uploaded data: " + ", ".join(runnable_types))
    else:
        st.warning("No ad type has a complete required upload set yet.")

if sp_ready and tacos_ready:
    try:
        with st.spinner("Reading diagnostic preview..."):
            diagnostics = safe_dict(build_engine().analyze())
        account_health = safe_dict(diagnostics.get("account_health"))
        account_summary = safe_dict(diagnostics.get("account_summary"))
        preview = safe_dict(diagnostics.get("pre_run_preview"))
        smart_warnings = safe_list(diagnostics.get("smart_warnings"))
        optimization_suggestions = safe_list(diagnostics.get("optimization_suggestions"))
        campaign_health_dashboard = safe_df(diagnostics.get("campaign_health_dashboard"))
        sqp_opportunities = safe_df(diagnostics.get("sqp_opportunities"))
        sqp_summary = safe_dict(diagnostics.get("sqp_summary"))
        top_opportunities = safe_df(diagnostics.get("top_opportunities"))
    except Exception as e:
        st.error(f"Diagnostics failed: {e}")
elif bulk_bytes is not None and not sp_ready:
    st.info("Upload the required Sponsored Products reports to unlock the full diagnostic preview.")


# =========================================================
# Diagnostic Preview
# =========================================================
if diagnostics:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sponsored Products Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">A quick health read before generating any changes.</div>', unsafe_allow_html=True)

    tacos_pct = account_health.get("tacos_pct")
    tacos_display = f"{tacos_pct}%" if tacos_pct is not None else ("Disabled" if not enable_tacos_control else "Missing")
    dh1, dh2, dh3, dh4, dh5, dh6 = st.columns(6)
    with dh1:
        render_metric_card("SP Ad Spend", f"${get_number(account_summary.get('total_spend')):,.2f}", tone="brand")
    with dh2:
        render_metric_card("SP Ad Sales", f"${get_number(account_summary.get('total_sales')):,.2f}", tone="brand")
    with dh3:
        roas = get_number(account_health.get("account_roas"))
        target = get_number(account_health.get("adjusted_min_roas"), min_roas)
        render_metric_card("SP Account ROAS", f"{roas:.2f}", tone="good" if roas >= target else "bad")
    with dh4:
        render_metric_card("TACOS", str(tacos_display), tone="warn", small=True)
    with dh5:
        render_metric_card("Under Target", str(get_int(account_summary.get("campaigns_under_target"))), tone="bad")
    with dh6:
        render_metric_card("Scalable", str(get_int(account_summary.get("campaigns_scalable"))), tone="good")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Smart Warnings & Suggestions</div>', unsafe_allow_html=True)
    sw1, sw2 = st.columns(2)
    with sw1:
        st.markdown("**Warnings**")
        if smart_warnings:
            for item in smart_warnings:
                st.markdown(f"- {item}")
        else:
            st.info("No warnings generated.")
    with sw2:
        st.markdown("**Suggestions**")
        if optimization_suggestions:
            for item in optimization_suggestions:
                st.markdown(f"- {item}")
        else:
            st.info("No suggestions generated.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pre-Run Action Preview</div>', unsafe_allow_html=True)
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

    if not top_opportunities.empty:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Scaling Opportunities</div>', unsafe_allow_html=True)
        display_df(top_opportunities, preferred=["campaign_name", "ad_group_name", "target", "match_type", "brand_segment", "placement_bucket", "roas", "orders", "clicks", "impression_share_pct", "score", "confidence", "reason"], height=420)

    if not campaign_health_dashboard.empty:
        with st.expander("Campaign Health Table", expanded=False):
            display_df(campaign_health_dashboard, height=420)

    if sqp_summary.get("uploaded"):
        with st.expander("Top SQP Opportunities", expanded=False):
            display_df(sqp_opportunities, preferred=["search_query", "search_query_score", "search_query_volume", "purchase_rate_pct", "purchases_total_count", "purchases_brand_share_pct", "opportunity_tier", "recommended_action", "in_search_term_report"], height=420)

    if enable_ai_review and api_key_present:
        render_info_banner("AI optimization is enabled", "After you run the optimizer, AI will automatically review low-confidence actions and use SQP context if available.", tone="brand")


# =========================================================
# Run Optimizer
# =========================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Run Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">When your uploads and settings are ready, run the optimizer across all available ad types.</div>', unsafe_allow_html=True)
run_left, run_center, run_right = st.columns([2, 3, 2])
with run_center:
    run_optimizer = st.button("Run Optimization for Available Ad Types", type="primary", use_container_width=True)
    st.caption("This will apply the current settings, guardrails, and uploaded report context.")

if run_optimizer:
    if not runnable_types:
        st.error("Please upload the required report set for at least one ad type.")
    elif enable_tacos_control and business_bytes is None and sp_ready:
        st.error("Please upload a Sales and Traffic Business Report to use TACOS control.")
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
    if not combined_bulk_updates.empty:
        combined_bulk_updates.columns = [str(c).strip() for c in combined_bulk_updates.columns]
        combined_bulk_updates = combined_bulk_updates.loc[:, ~pd.Index(combined_bulk_updates.columns).duplicated(keep="first")].copy()

    sp_bulk_updates = product_bulk_slice(combined_bulk_updates, "Sponsored Products")
    sb_bulk_updates = product_bulk_slice(combined_bulk_updates, "Sponsored Brands")
    sd_bulk_updates = product_bulk_slice(combined_bulk_updates, "Sponsored Display")
    bid_recommendations = safe_df(outputs.get("bid_recommendations"))
    search_term_actions = safe_df(outputs.get("search_term_actions"))
    campaign_budget_actions = safe_df(outputs.get("campaign_budget_actions"))
    output_account_health = safe_dict(outputs.get("account_health"))
    simulation_summary = safe_dict(outputs.get("simulation_summary"))
    run_history = safe_df(outputs.get("run_history"))
    output_account_summary = safe_dict(outputs.get("account_summary"))
    output_sqp_opportunities = safe_df(outputs.get("sqp_opportunities"))
    output_sqp_summary = safe_dict(outputs.get("sqp_summary"))
    execution_summary = safe_df(outputs.get("execution_summary"))
    per_type_outputs = safe_dict(outputs.get("per_type_outputs"))
    output_runnable_types = safe_list(outputs.get("runnable_types"))
    optimizer_diagnostics = safe_df(outputs.get("optimizer_diagnostics"))
    output_top_opportunities = safe_df(outputs.get("top_opportunities"))

    placement_summary_df = summarize_placement_actions(bid_recommendations)
    brand_segment_summary_df = summarize_brand_segments(bid_recommendations, search_term_actions)
    trend_memory_summary_df = summarize_trend_memory(bid_recommendations, campaign_budget_actions)
    portfolio_reallocation_summary_df = summarize_portfolio_reallocation(campaign_budget_actions)

    tacos_total_ad_spend_used = get_number(output_account_health.get("effective_total_ad_spend"))
    tacos_business_sales_used = get_number(output_account_health.get("business_total_sales"))
    tacos_pct_value = output_account_health.get("tacos_pct")
    tacos_status_value = str(output_account_health.get("tacos_status", "not_used")).replace("_", " ").title()
    tacos_guardrail_target = max_tacos_target if enable_tacos_control else None

    ai_override_log_df = pd.DataFrame()
    ai_impact_summary = {}
    ai_executive_summary = ""
    original_bulk_count = len(combined_bulk_updates)

    if enable_ai_review and api_key_present and not combined_bulk_updates.empty:
        try:
            ai_candidates_df = build_ai_action_candidates(combined_bulk_updates, bid_recommendations, search_term_actions, campaign_budget_actions)
            low_conf_df = ai_candidates_df[ai_candidates_df["confidence"] == "LOW"].copy().head(25)
            if not low_conf_df.empty:
                payload = build_ai_override_payload(low_conf_df, output_account_summary, output_account_health, simulation_summary, output_sqp_opportunities, output_sqp_summary)
                ai_response = run_ai_review_cached(payload_json=json.dumps(payload, default=str))
                if ai_response.get("_model_used"):
                    st.caption(f"AI model used: {ai_response['_model_used']}")
                ai_executive_summary = str(ai_response.get("executive_summary", "")).strip()
                combined_bulk_updates, ai_override_log_df = apply_ai_overrides_to_combined(combined_bulk_updates, ai_response, ai_candidates_df, strategy_mode, max_bid_cap, max_budget_cap)
                ai_impact_summary = build_ai_impact_summary(ai_override_log_df, original_bulk_count, len(combined_bulk_updates), ai_executive_summary)
                sp_bulk_updates = product_bulk_slice(combined_bulk_updates, "Sponsored Products")
                sb_bulk_updates = product_bulk_slice(combined_bulk_updates, "Sponsored Brands")
                sd_bulk_updates = product_bulk_slice(combined_bulk_updates, "Sponsored Display")
        except Exception as e:
            st.warning(f"AI optimization skipped: {e}")

    suppressed_summary = estimate_tacos_suppressed_scaling(combined_bulk_updates, output_account_health, simulation_summary)
    conflicting_signals = build_conflicting_signals(output_account_health, simulation_summary, output_account_summary)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Execution Summary</div>', unsafe_allow_html=True)
    es1, es2, es3 = st.columns(3)
    with es1:
        render_metric_card("Runnable Types", str(len(output_runnable_types)), tone="good" if output_runnable_types else "warn")
    with es2:
        render_metric_card("Bulk Actions", str(len(combined_bulk_updates)), tone="brand")
    with es3:
        render_metric_card("Ad Types with Output", str(len(per_type_outputs)), tone="brand")
    display_df(execution_summary, height=260)

    if not optimizer_diagnostics.empty:
        with st.expander("Optimizer Diagnostics", expanded=False):
            display_df(optimizer_diagnostics, height=320)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Optimization Summary</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Simulation & Confidence</div>', unsafe_allow_html=True)
    sim1, sim2, sim3 = st.columns(3)
    with sim1:
        render_metric_card("High Confidence", str(get_int(simulation_summary.get("high_confidence_actions"))), tone="good")
    with sim2:
        render_metric_card("Low Confidence", str(get_int(simulation_summary.get("low_confidence_actions"))), tone="warn")
    with sim3:
        render_metric_card("Estimated Spend Impact", f"{get_number(simulation_summary.get('estimated_spend_impact_pct')):,.2f}%", tone="brand", small=True)

    if enable_tacos_control:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">TACOS Calculation Transparency</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">This shows the exact numerator and denominator used for the account-wide TACOS guardrail.</div>', unsafe_allow_html=True)
        tx1, tx2, tx3, tx4 = st.columns(4)
        with tx1:
            render_metric_card("Total Ad Spend Used", f"${tacos_total_ad_spend_used:,.2f}", tone="brand")
        with tx2:
            render_metric_card("Business Report Sales Used", f"${tacos_business_sales_used:,.2f}" if tacos_business_sales_used > 0 else "Missing", tone="brand" if tacos_business_sales_used > 0 else "bad")
        with tx3:
            tacos_tone = "bad" if str(output_account_health.get("tacos_status", "")).lower() == "above_target" else "good"
            render_metric_card("Account-Wide TACOS", f"{float(tacos_pct_value):.2f}%" if tacos_pct_value is not None else "Missing", tone=tacos_tone)
        with tx4:
            render_metric_card("Max TACOS Target", f"{tacos_guardrail_target:.2f}%" if tacos_guardrail_target is not None else "Disabled", tone="warn")
        st.caption(f"Scaling Status: {'Constrained' if str(output_account_health.get('tacos_status', '')).lower() == 'above_target' else 'Allowed'} ({tacos_status_value})")

        sx1, sx2, sx3 = st.columns(3)
        with sx1:
            render_metric_card("Suppressed Bid Increases", str(suppressed_summary.get("suppressed_bid_increases", 0)), tone="warn")
        with sx2:
            render_metric_card("Suppressed Budget Increases", str(suppressed_summary.get("suppressed_budget_increases", 0)), tone="warn")
        with sx3:
            render_metric_card("Total Scaling Suppressed", str(suppressed_summary.get("suppressed_total", 0)), tone="warn")

    if conflicting_signals:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Conflicting Signals</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">These are important situations where performance signals and account guardrails are pulling in different directions.</div>', unsafe_allow_html=True)
        for msg in conflicting_signals:
            st.warning(msg)

    if not placement_summary_df.empty:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Placement Optimization Reporting</div>', unsafe_allow_html=True)
        render_info_banner("Placement-aware decisions detected", "This summary shows how top of search, product pages, and rest of search are being treated in the current recommendation set.", tone="good")
        display_df(placement_summary_df, height=280)

    if not brand_segment_summary_df.empty:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Branded vs Non-Branded Reporting</div>', unsafe_allow_html=True)
        display_df(brand_segment_summary_df, height=280)

    if not trend_memory_summary_df.empty:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Trend Memory & Anti-Oscillation Reporting</div>', unsafe_allow_html=True)
        display_df(trend_memory_summary_df, height=280)

    if not portfolio_reallocation_summary_df.empty:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Portfolio Budget Reallocation Reporting</div>', unsafe_allow_html=True)
        display_df(portfolio_reallocation_summary_df, height=280)

    if not output_top_opportunities.empty:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Opportunities</div>', unsafe_allow_html=True)
        display_df(output_top_opportunities, preferred=["campaign_name", "ad_group_name", "target", "match_type", "brand_segment", "placement_bucket", "roas", "orders", "clicks", "impression_share_pct", "score", "confidence", "reason"], height=420)

    if output_sqp_summary.get("uploaded"):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">SQP Opportunity Reporting</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Optimization Narrative</div>', unsafe_allow_html=True)
    for point in build_narrative(output_account_health, simulation_summary, pacing_status):
        st.markdown(f"- {point}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Results Explorer</div>', unsafe_allow_html=True)
    tab_names = [
        "Execution Summary",
        "Bulk Upload",
        "Bid Recommendations",
        "Search Term Actions",
        "Budget Actions",
        "Placement",
        "Brand Segments",
        "Trend Memory",
        "Portfolio Reallocation",
        "Top Opportunities",
        "SQP Opportunities",
        "AI Override Log",
        "Diagnostics",
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        display_df(execution_summary, height=520)
    with tabs[1]:
        display_df(combined_bulk_updates, preferred=["Product", "Entity", "Campaign Name", "Ad Group Name", "Keyword Text", "Match Type", "Bid", "Budget", "Daily Budget", "Optimizer Action", "Confidence", "Score", "Reason"], height=520)
    with tabs[2]:
        display_df(bid_recommendations, preferred=["campaign_name", "ad_group_name", "target", "match_type", "placement_bucket", "brand_segment", "current_bid", "recommended_bid", "recommended_action", "clicks", "orders", "roas", "impression_share_pct", "score", "confidence", "reason"], height=520)
    with tabs[3]:
        display_df(search_term_actions, preferred=["campaign_name", "ad_group_name", "search_term", "match_type", "placement_bucket", "brand_segment", "search_term_action", "recommended_bid", "clicks", "orders", "roas", "score", "confidence", "reason"], height=520)
    with tabs[4]:
        display_df(campaign_budget_actions, preferred=["campaign_name", "portfolio_name", "portfolio_key", "reallocation_role", "daily_budget", "recommended_daily_budget", "reallocation_delta", "campaign_action", "clicks", "orders", "roas", "avg_impression_share_pct", "score", "confidence", "reason"], height=520)
    with tabs[5]:
        display_df(placement_summary_df, height=520)
    with tabs[6]:
        display_df(brand_segment_summary_df, height=520)
    with tabs[7]:
        display_df(trend_memory_summary_df, height=520)
    with tabs[8]:
        display_df(portfolio_reallocation_summary_df, height=520)
    with tabs[9]:
        display_df(output_top_opportunities, preferred=["campaign_name", "ad_group_name", "target", "match_type", "brand_segment", "placement_bucket", "roas", "orders", "clicks", "impression_share_pct", "score", "confidence", "reason"], height=520)
    with tabs[10]:
        display_df(output_sqp_opportunities, preferred=["search_query", "search_query_score", "search_query_volume", "purchase_rate_pct", "purchases_total_count", "purchases_brand_share_pct", "opportunity_tier", "recommended_action", "in_search_term_report"], height=520)
    with tabs[11]:
        display_df(ai_override_log_df, height=520)
    with tabs[12]:
        display_df(optimizer_diagnostics, height=520)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Downloads</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Export the primary bulk uploads below. Additional reports are available in the optional section.</div>', unsafe_allow_html=True)

    primary1, primary2, primary3 = st.columns(3)
    with primary1:
        if not sp_bulk_updates.empty:
            st.download_button("Download SP Bulk Upload", data=to_excel_bytes(sp_bulk_updates), file_name="amazon_bulk_updates_sp.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        else:
            st.info("No SP bulk upload")
    with primary2:
        if not sb_bulk_updates.empty:
            st.download_button("Download SB Bulk Upload", data=to_excel_bytes(sb_bulk_updates), file_name="amazon_bulk_updates_sb.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        else:
            st.info("No SB bulk upload")
    with primary3:
        if not sd_bulk_updates.empty:
            st.download_button("Download SD Bulk Upload", data=to_excel_bytes(sd_bulk_updates), file_name="amazon_bulk_updates_sd.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        else:
            st.info("No SD bulk upload")

    if sp_bulk_updates.empty and sb_bulk_updates.empty and sd_bulk_updates.empty:
        st.info("No bulk-upload rows available for this run.")

    with st.expander("Optional Downloads", expanded=False):
        downloadable = {
            "Execution Summary": execution_summary,
            "Bid Recommendations": bid_recommendations,
            "Search Term Actions": search_term_actions,
            "Campaign Budget Actions": campaign_budget_actions,
            "Top Opportunities": output_top_opportunities,
            "SQP Opportunities": output_sqp_opportunities,
            "AI Override Log": ai_override_log_df,
            "Placement Summary": placement_summary_df,
            "Brand Segment Summary": brand_segment_summary_df,
            "Trend Memory Summary": trend_memory_summary_df,
            "Portfolio Reallocation Summary": portfolio_reallocation_summary_df,
        }
        for label, df in downloadable.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                key_name = normalize_text(label).replace(" ", "_")
                st.download_button(
                    f"Download {label}",
                    data=to_excel_bytes(df),
                    file_name=f"{key_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"download_{key_name}",
                )

    with st.expander("Run History", expanded=False):
        if not run_history.empty and "timestamp" in run_history.columns:
            display_df(run_history.sort_values(by="timestamp", ascending=False), height=320)
        else:
            display_df(run_history, height=320)


# =========================================================
# Footer
# =========================================================
footer_col1, footer_col2, footer_col3 = st.columns([3, 2, 3])
with footer_col2:
    st.markdown('<div class="footer-note">Evolved Commerce Amazon Ads Command Center</div>', unsafe_allow_html=True)
