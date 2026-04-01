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


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Output")
    return output.getvalue()


def safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def get_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
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
    seen = set()
    ordered = []
    for model in preferred + fallbacks:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def normalize_match_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_term_text(value: Any) -> str:
    value = str(value or "").lower()
    value = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in value)
    return " ".join(value.split())


def score_action_confidence(row: dict) -> str:
    clicks = float(row.get("clicks", 0) or 0)
    orders = float(row.get("orders", 0) or 0)
    roas = float(row.get("roas", 0) or 0)
    source_type = str(row.get("source_type", "")).lower()
    action = str(row.get("optimizer_action", "")).upper()

    if source_type in {"graduation", "product_target"} and orders >= 2:
        return "MEDIUM"
    if action == "ADD_NEGATIVE_EXACT" and clicks > 10:
        return "HIGH"
    if clicks >= 20 and orders == 0:
        return "HIGH"
    if orders >= 3 and roas >= 3:
        return "HIGH"
    if source_type == "budget" and clicks < 25:
        return "LOW"
    if source_type == "bid" and clicks < 8:
        return "LOW"
    return "MEDIUM"


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
            notes.append("Monthly pacing is currently over target, so scaling was suppressed.")
        elif pacing_status == "On Pace":
            notes.append("Monthly pacing is on target, so standard optimization behavior was allowed.")

    if simulation_summary.get("bid_decreases", 0) > 0:
        notes.append(f"{simulation_summary['bid_decreases']} bid decreases were generated to reduce waste.")
    if simulation_summary.get("bid_increases", 0) > 0:
        notes.append(f"{simulation_summary['bid_increases']} bid increases were generated for strong-performing targets.")
    found_graduations = simulation_summary.get("graduation_opportunities", simulation_summary.get("graduations", 0))
    ready_graduations = simulation_summary.get("graduations", 0)
    
    if found_graduations > 0:
        if found_graduations == ready_graduations:
            notes.append(
                f"{ready_graduations} search terms or ASINs were graduated into Dest or Research structures."
            )
        else:
            notes.append(
                f"{found_graduations} graduation opportunities were found, and {ready_graduations} were ready for this upload."
            )
    if simulation_summary.get("negatives_added", 0) > 0:
        notes.append(f"{simulation_summary['negatives_added']} Negative Exact rows were created for losers or source cleanup.")
    if simulation_summary.get("campaign_creates", 0) > 0:
        notes.append(f"{simulation_summary['campaign_creates']} campaign creation row(s) were generated.")
    if simulation_summary.get("budget_increases", 0) > 0 or simulation_summary.get("budget_decreases", 0) > 0:
        notes.append(
            f"Campaign budget actions included {simulation_summary.get('budget_increases', 0)} increases and {simulation_summary.get('budget_decreases', 0)} decreases."
        )
    if tacos_pct is not None:
        notes.append(f"Current TACOS reading from this run is {tacos_pct}%.")

    return notes


@st.cache_data(show_spinner=False, ttl=900)
def run_ai_review_cached(payload_json: str) -> dict:
    client = get_openai_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY not found.")

    instructions = """
You are an expert Amazon Sponsored Products optimizer.

You are reviewing ONLY low-confidence optimization actions.
Be conservative. Prefer KEEP over MODIFY.
Do not invent data.

Allowed MODIFY actions:
- bid rows: INCREASE_BID, DECREASE_BID, NO_ACTION
- budget rows: INCREASE_BUDGET, DECREASE_BUDGET, NO_ACTION
- graduation rows: ADD_TO_DEST_EXACT, ADD_TO_RESEARCH_PHRASE, ADD_ASIN_TO_DEST, NO_ACTION
- negative rows: ADD_NEGATIVE_EXACT, NO_ACTION
- campaign/ad group create rows: KEEP or REMOVE only

If decision is KEEP, set new_action to an empty string.
If decision is REMOVE, set new_action to an empty string.
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
                max_output_tokens=1600,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ai_overrides",
                        "schema": schema,
                    }
                },
            )
            raw_text = getattr(response, "output_text", "") or ""
            if not raw_text:
                raise ValueError("No readable AI output.")
            parsed = json.loads(raw_text)
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
        bid["_campaign_key"] = bid["campaign_name"].map(normalize_match_text)
        bid["_ad_group_key"] = bid["ad_group_name"].map(normalize_match_text)
        bid["_term_key"] = bid["target"].map(normalize_term_text)
        bid["_match_key"] = bid["match_type"].map(normalize_match_text)

    if not search.empty:
        search["_campaign_key"] = search["campaign_name"].map(normalize_match_text)
        search["_ad_group_key"] = search["ad_group_name"].map(normalize_match_text)
        search["_term_key"] = search["normalized_term"].map(normalize_term_text)
        search["_asin_key"] = search["asin_value"].fillna("").astype(str).map(normalize_term_text)
        search["_action_key"] = search["search_term_action"].astype(str).str.upper()

    if not budget.empty:
        budget["_campaign_key"] = budget["campaign_name"].map(normalize_match_text)
        budget["_action_key"] = budget["campaign_action"].astype(str).str.upper()

    rows = []
    for _, row in combined.iterrows():
        optimizer_action = str(row.get("Optimizer Action", "")).upper()
        entity = str(row.get("Entity", ""))
        keyword_text = str(row.get("Keyword Text", ""))
        product_expr = str(row.get("Product Targeting Expression", ""))

        candidate = {
            "id": str(row.get("id", "")),
            "source_type": "unknown",
            "optimizer_action": optimizer_action,
            "campaign_name": str(row.get("Campaign Name", "")),
            "ad_group_name": str(row.get("Ad Group Name", "")),
            "keyword_text": keyword_text or product_expr,
            "match_type": str(row.get("Match Type", "")),
            "entity": entity,
            "clicks": 0.0,
            "orders": 0.0,
            "spend": 0.0,
            "sales": 0.0,
            "roas": 0.0,
            "current_bid": None,
            "recommended_bid": None,
            "current_daily_budget": None,
            "recommended_daily_budget": None,
        }

        campaign_key = normalize_match_text(row.get("Campaign Name", ""))
        ad_group_key = normalize_match_text(row.get("Ad Group Name", ""))
        term_key = normalize_term_text(keyword_text)
        product_key = normalize_term_text(product_expr)
        match_key = normalize_match_text(row.get("Match Type", ""))

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
                        "current_bid": get_number(m.get("current_bid")),
                        "recommended_bid": get_number(m.get("recommended_bid")),
                    }
                )

        elif optimizer_action in {"ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST", "ADD_NEGATIVE_EXACT", "NEGATE_SOURCE_EXISTING_DEST"} and not search.empty:
            match = search[
                (search["_campaign_key"] == campaign_key)
                & (search["_ad_group_key"] == ad_group_key)
                & (
                    (search["_term_key"] == term_key)
                    | (search["_asin_key"] == term_key)
                    | (search["_asin_key"] == product_key)
                )
            ]
            if not match.empty:
                m = match.iloc[0]
                candidate.update(
                    {
                        "source_type": "product_target" if optimizer_action == "ADD_ASIN_TO_DEST" else "graduation",
                        "clicks": get_number(m.get("clicks")),
                        "orders": get_number(m.get("orders")),
                        "spend": get_number(m.get("spend")),
                        "sales": get_number(m.get("sales")),
                        "roas": get_number(m.get("roas")),
                        "recommended_bid": get_number(m.get("recommended_bid")),
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
                        "current_daily_budget": get_number(m.get("daily_budget")),
                        "recommended_daily_budget": get_number(m.get("recommended_daily_budget")),
                    }
                )

        elif optimizer_action in {"CREATE_CAMPAIGN", "CREATE_AD_GROUP"}:
            candidate["source_type"] = "structure_create"

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if combined_bulk_updates.empty:
        return combined_bulk_updates.copy(), pd.DataFrame()

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
        was_removed = False

        override = override_map.get(rid)
        if override:
            decision = str(override.get("decision", "KEEP")).upper()
            reason = str(override.get("reason", "")).strip()
            new_action = str(override.get("new_action", "")).upper().strip()

            if decision == "REMOVE" or new_action == "NO_ACTION":
                was_removed = True
            elif decision == "MODIFY" and new_action:
                final_action = new_action
                row_dict["Optimizer Action"] = final_action

                if final_action == "ADD_NEGATIVE_EXACT":
                    row_dict["Entity"] = "Negative Keyword"
                    row_dict["Operation"] = "Create"
                    row_dict["Match Type"] = "Negative Exact"
                elif final_action in {"ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE"}:
                    row_dict["Entity"] = "Keyword"
                    row_dict["Operation"] = "Create"
                    row_dict["Match Type"] = "Exact" if final_action == "ADD_TO_DEST_EXACT" else "Phrase"
                elif final_action == "ADD_ASIN_TO_DEST":
                    row_dict["Entity"] = "Product Targeting"
                    row_dict["Operation"] = "Create"

        if not was_removed:
            updated_rows.append(row_dict)

        candidate = candidate_map.get(rid, {})
        log_rows.append(
            {
                "ID": rid,
                "Source Type": candidate.get("source_type", "unknown"),
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

    return pd.DataFrame(updated_rows), pd.DataFrame(log_rows)


def build_ai_impact_summary(
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
            "original_count": original_count,
            "final_count": final_count,
            "executive_summary": executive_summary,
        }

    return {
        "reviewed": int(len(ai_override_log_df)),
        "kept": int((ai_override_log_df["Decision"] == "KEEP").sum()),
        "modified": int((ai_override_log_df["Decision"] == "MODIFY").sum()),
        "removed": int((ai_override_log_df["Final Action"] == "REMOVED").sum()),
        "original_count": original_count,
        "final_count": final_count,
        "executive_summary": executive_summary,
    }


st.markdown(
    """
    <style>
        .main > div { padding-top: 1.1rem; }
        .block-container { padding-top: 1.1rem; padding-bottom: 1.75rem; max-width: 1440px; }
        [data-testid="stSidebar"] { background: #F3F4F6; }

        .brand-shell {
            background: linear-gradient(135deg, #EA580C 0%, #1F2937 100%);
            border-radius: 20px;
            padding: 65px 30px;
            color: white;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
        }

        .brand-title { font-size: 2.15rem; font-weight: 800; line-height: 1.05; margin: 0; }
        .brand-subtitle { font-size: 1rem; opacity: 0.96; margin-top: 0.55rem; }
        .section-title { font-size: 1.18rem; font-weight: 750; color: #111827; margin-bottom: 0.2rem; }
        .section-note { color: #4B5563; font-size: 0.94rem; margin-bottom: 0.9rem; }
        .section-divider { border-top: 1px solid #E5E7EB; margin: 1.1rem 0 1.15rem 0; }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            min-height: 80px;
        }

        .metric-label { font-size: 0.75rem; color: #6B7280; margin-bottom: 0.25rem; font-weight: 600; }
        .metric-value { font-size: 1.5rem; line-height: 1.1; font-weight: 800; color: #1F2937; word-break: break-word; }
        .metric-value.small { font-size: 1.15rem; }
        .metric-card.good { border-left: 4px solid #10B981; }
        .metric-card.warn { border-left: 4px solid #F59E0B; }
        .metric-card.bad { border-left: 4px solid #EF4444; }
        .metric-card.brand { border-left: 4px solid #F47322; }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## Tool Guide")
    st.markdown(
        """
**What this tool does**
- Adjusts bids
- Graduates winning search terms into Dest / Research structures
- Adds Negative Exacts for losers and source cleanup
- Creates missing Dest / Research campaigns and ad groups
- Routes ASIN winners into `ASIN Targets`
- Adjusts campaign daily budgets
- Optionally enforces TACOS and monthly pacing guardrails
- Optionally uses prior-month SQP Simple View for context
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
- Search Query Performance Report (Simple View, prior month)
"""
    )

    st.markdown("---")
    enable_ai_review = st.checkbox("Enable AI Optimization Layer", value=True)
    api_key_present = bool(get_openai_api_key())
    if enable_ai_review and not api_key_present:
        st.warning("OPENAI_API_KEY not found. AI review will be skipped.")

    st.markdown("---")
    st.markdown("**Version:** 2.0")
    st.markdown("**Owner:** Jake Yorgason, Evolved Commerce")


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
                Bid optimization • Graduation engine • Negative Exact cleanup • Budget pacing • TACOS control • SQP context • AI optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Optimization Settings</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Core controls for bidding, efficiency, graduation, and cleanup logic.</div>', unsafe_allow_html=True)

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    min_roas = st.number_input("Minimum ROAS Target", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
with r1c2:
    min_clicks = st.number_input("Minimum Clicks Before Standard Efficiency Actions", min_value=1, max_value=100, value=8, step=1)
with r1c3:
    strategy_mode = st.selectbox("Strategy Mode", options=["Conservative", "Balanced", "Aggressive"], index=1)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

with st.expander("Graduation Rules", expanded=False):
    st.markdown(
        """
Configure how the optimizer decides whether a search term should:
- graduate into Dest Exact
- route into Research Phrase
- be negated as a loser
- or create missing destination structure
"""
    )

    g1, g2 = st.columns(2)
    with g1:
        min_orders_for_graduation = st.number_input(
            "Min Orders for Graduation",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="Minimum attributed orders required before a search term or ASIN can be considered for graduation."
        )
        dest_acos_threshold = st.number_input(
            "Dest ACOS % Threshold",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=0.5,
            help="Terms at or below this ACOS threshold qualify for Dest Exact when they also meet the order rule."
        )
        create_missing_dest_campaigns = st.checkbox(
            "Create Missing Dest / Research",
            value=True,
            help="If enabled, the optimizer will output campaign/ad group structure rows when the required Dest or Research destination does not already exist."
        )

    with g2:
        new_target_bid_multiplier = st.number_input(
            "New Target Bid Multiplier",
            min_value=1.0,
            max_value=3.0,
            value=1.10,
            step=0.01,
            help="Starting bid for new graduated keywords/ASIN targets is current CPC multiplied by this value."
        )
        new_target_bid_cap = st.number_input(
            "New Target Bid Cap",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.05,
            help="Maximum starting bid allowed for newly created Dest / Research / ASIN targets."
        )

    st.markdown("#### Research routing")
    r1, r2 = st.columns(2)
    with r1:
        research_ctr_low_pct = st.number_input(
            "Research CTR Low %",
            min_value=0.01,
            max_value=5.0,
            value=0.10,
            step=0.01,
            help="Lower CTR bound for routing a converting term into Research Phrase instead of Dest Exact."
        )
        research_ctr_high_pct = st.number_input(
            "Research CTR High %",
            min_value=0.01,
            max_value=5.0,
            value=0.25,
            step=0.01,
            help="Upper CTR bound for Research Phrase routing."
        )

    with r2:
        research_cvr_low_pct = st.number_input(
            "Research CVR Low %",
            min_value=0.1,
            max_value=20.0,
            value=2.0,
            step=0.1,
            help="Lower conversion-rate bound for Research Phrase routing."
        )
        research_cvr_high_pct = st.number_input(
            "Research CVR High %",
            min_value=0.1,
            max_value=20.0,
            value=5.0,
            step=0.1,
            help="Upper conversion-rate bound for Research Phrase routing."
        )

    st.markdown("#### Loser cleanup")
    l1, l2 = st.columns(2)
    with l1:
        loser_clicks_threshold = st.number_input(
            "Loser Click Threshold",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            help="Minimum clicks before a low-quality term becomes eligible for Negative Exact cleanup."
        )
        loser_ctr_threshold_pct = st.number_input(
            "Loser CTR Max %",
            min_value=0.01,
            max_value=10.0,
            value=0.25,
            step=0.01,
            help="Terms below this CTR and below the CVR threshold can be negated once the click threshold is met."
        )

    with l2:
        loser_cvr_threshold_pct = st.number_input(
            "Loser CVR Max %",
            min_value=0.1,
            max_value=20.0,
            value=5.0,
            step=0.1,
            help="Terms below this conversion rate and below the CTR threshold can be negated once the click threshold is met."
        )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Optimization Actions</div>', unsafe_allow_html=True)

core1, core2, core3, core4 = st.columns(4)
with core1:
    enable_bid_updates = st.checkbox("Bid Updates", value=True)
with core2:
    enable_search_harvesting = st.checkbox("Graduation Engine", value=True)
with core3:
    enable_negative_keywords = st.checkbox("Negative Exact Cleanup", value=True)
with core4:
    enable_budget_updates = st.checkbox("Budget Updates", value=True)

with st.expander("Efficiency Guardrails", expanded=False):
    eg1, eg2 = st.columns(2)
    with eg1:
        enable_tacos_control = st.checkbox("Enable TACOS Control", value=False)
    with eg2:
        max_tacos_target = st.number_input("Maximum TACOS %", min_value=1.0, max_value=100.0, value=15.0, step=0.5)

with st.expander("Budget Guardrails", expanded=False):
    bg1, bg2 = st.columns(2)
    with bg1:
        enable_monthly_budget_control = st.checkbox("Enable Monthly Budget Control", value=False)
    with bg2:
        pacing_buffer_pct = st.number_input("Pacing Buffer %", min_value=0.0, max_value=50.0, value=5.0, step=1.0)

    bg3, bg4 = st.columns(2)
    with bg3:
        monthly_account_budget = st.number_input("Monthly Account Budget", min_value=0.0, max_value=1000000.0, value=0.0, step=100.0)
    with bg4:
        month_to_date_spend = st.number_input("Month-to-Date Spend", min_value=0.0, max_value=1000000.0, value=0.0, step=100.0)

with st.expander("Caps", expanded=False):
    cap1, cap2 = st.columns(2)
    with cap1:
        max_bid_cap = st.number_input("Maximum Recommended Bid", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    with cap2:
        max_budget_cap = st.number_input("Maximum Recommended Daily Budget", min_value=1.0, max_value=10000.0, value=500.0, step=1.0)

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

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload Amazon Reports</div>', unsafe_allow_html=True)

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
    business_file = st.file_uploader("Sales and Traffic Business Report", type=["xlsx", "csv"])
    if business_file is not None:
        st.success("Seller Central Business Report uploaded")
    sqp_file = st.file_uploader("Search Query Performance Report (Simple View, optional)", type=["csv"])
    if sqp_file is not None:
        st.success("SQP report uploaded")

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
        zero_order_click_threshold=loser_clicks_threshold,
        zero_order_action="Both",
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
        enable_dest_graduation=True,
        enable_research_graduation=True,
        enable_asin_graduation=True,
        create_missing_dest_campaigns=create_missing_dest_campaigns,
        create_missing_research_campaigns=create_missing_dest_campaigns,
        min_orders_for_graduation=min_orders_for_graduation,
        dest_acos_threshold=dest_acos_threshold,
        research_ctr_low=research_ctr_low_pct / 100,
        research_ctr_high=research_ctr_high_pct / 100,
        research_cvr_low=research_cvr_low_pct / 100,
        research_cvr_high=research_cvr_high_pct / 100,
        loser_clicks_threshold=loser_clicks_threshold,
        loser_ctr_threshold=loser_ctr_threshold_pct / 100,
        loser_cvr_threshold=loser_cvr_threshold_pct / 100,
        new_target_bid_multiplier=new_target_bid_multiplier,
        new_target_bid_cap=new_target_bid_cap,
    )


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
        account_health = safe_dict(diagnostics.get("account_health"))
        account_summary = safe_dict(diagnostics.get("account_summary"))
        preview = safe_dict(diagnostics.get("pre_run_preview"))
        smart_warnings = diagnostics.get("smart_warnings", [])
        optimization_suggestions = diagnostics.get("optimization_suggestions", [])
        campaign_health_dashboard = safe_df(diagnostics.get("campaign_health_dashboard"))
        sqp_opportunities = safe_df(diagnostics.get("sqp_opportunities"))
        sqp_summary = safe_dict(diagnostics.get("sqp_summary"))
    except Exception as e:
        st.error(f"Diagnostics failed: {e}")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What This Run Is Allowed To Do</div>', unsafe_allow_html=True)

st.markdown("- Adjust bids on existing keyword / target rows.")
st.markdown("- Add proven winners to **Dest Exact** when orders and ACOS qualify.")
st.markdown("- Route borderline winners to **Research Phrase**.")
st.markdown("- Route ASIN winners to a regular Dest campaign under **ASIN Targets**.")
st.markdown("- Add **Negative Exact** for losers and source cleanup after graduation.")
st.markdown("- Create missing Dest / Research campaigns and ad groups when routing is reliable.")
st.markdown("- Update campaign daily budgets at the campaign level only.")
if enable_tacos_control:
    st.markdown(f"- If account TACOS rises above **{max_tacos_target:.1f}%**, the app tightens scaling.")
if enable_monthly_budget_control and monthly_account_budget > 0:
    st.markdown("- Monthly budget pacing can suppress budget and bid increases.")
if enable_ai_review:
    st.markdown("- AI optimization reviews only lower-confidence actions after the core engine finishes.")

if diagnostics:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Diagnostics</div>', unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        render_metric_card("Account ROAS", str(account_health.get("account_roas", 0)), "brand")
    with a2:
        tacos_display = f"{account_health.get('tacos_pct', 'N/A')}%" if account_health.get("tacos_pct") is not None else "N/A"
        render_metric_card("TACOS", tacos_display, "warn")
    with a3:
        render_metric_card("Total Spend", f"${account_summary.get('total_spend', 0):,.2f}", "good")
    with a4:
        render_metric_card("Total Sales", f"${account_summary.get('total_sales', 0):,.2f}", "good")

    st.markdown("**Smart warnings**")
    for item in smart_warnings:
        st.markdown(f"- {item}")

    st.markdown("**Suggestions**")
    for item in optimization_suggestions:
        st.markdown(f"- {item}")

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    with p1:
        render_metric_card("Bid Increases", str(preview.get("bid_increases", 0)), "good")
    with p2:
        render_metric_card("Bid Decreases", str(preview.get("bid_decreases", 0)), "warn")
    with p3:
        render_metric_card("Neg. Exacts", str(preview.get("negatives_added", 0)), "warn")
    with p4:
        render_metric_card("Graduations", str(preview.get("graduations", 0)), "good")
    with p5:
        render_metric_card("Campaign Creates", str(preview.get("campaign_creates", 0)), "brand")
    with p6:
        render_metric_card("Budget Actions", str(preview.get("budget_increases", 0) + preview.get("budget_decreases", 0)), "brand")

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

if "last_outputs" in st.session_state:
    outputs = safe_dict(st.session_state["last_outputs"])

    combined_bulk_updates = safe_df(outputs.get("combined_bulk_updates"))

    preview_summary = safe_dict(outputs.get("pre_run_preview"))
    
    simulation_summary["graduation_opportunities"] = int(preview_summary.get("graduations", 0))
    
    dest_structure_upload = combined_bulk_updates[
        combined_bulk_updates["Optimizer Action"].isin(["CREATE_CAMPAIGN", "CREATE_AD_GROUP"])
    ].copy()
    
    dest_structure_upload = dest_structure_upload[
        dest_structure_upload["Campaign Name"].fillna("").astype(str).str.contains("Dest|Destination|Research", case=False, regex=True)
    ].copy()
    
    deferred_graduations = search_term_actions[
        search_term_actions["search_term_action"].isin(["ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST"])
    ].copy()
    
    ready_terms = set(
        combined_bulk_updates.loc[
            combined_bulk_updates["Optimizer Action"].isin(["ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST"]),
            "Keyword Text"
        ].fillna("").astype(str).str.lower().str.strip().tolist()
    )
    
    ready_asins = set(
        combined_bulk_updates.loc[
            combined_bulk_updates["Optimizer Action"] == "ADD_ASIN_TO_DEST",
            "Product Targeting Expression"
        ].fillna("").astype(str).str.lower().str.strip().tolist()
    )
    
    deferred_graduations = deferred_graduations[
        ~(
            deferred_graduations["normalized_term"].fillna("").astype(str).str.lower().str.strip().isin(ready_terms)
            | deferred_graduations["product_expression"].fillna("").astype(str).str.lower().str.strip().isin(ready_asins)
        )
    ].copy()
    
    bid_recommendations = safe_df(outputs.get("bid_recommendations"))
    search_term_actions = safe_df(outputs.get("search_term_actions"))
    campaign_budget_actions = safe_df(outputs.get("campaign_budget_actions"))
    output_account_health = safe_dict(outputs.get("account_health"))
    output_account_summary = safe_dict(outputs.get("account_summary"))
    simulation_summary = safe_dict(outputs.get("simulation_summary"))
    preview_summary = safe_dict(outputs.get("pre_run_preview"))
    simulation_summary["graduation_opportunities"] = int(preview_summary.get("graduations", 0))
    run_history = safe_df(outputs.get("run_history"))
    output_campaign_health_dashboard = safe_df(outputs.get("campaign_health_dashboard"))
    output_sqp_opportunities = safe_df(outputs.get("sqp_opportunities"))
    output_sqp_summary = safe_dict(outputs.get("sqp_summary"))

    ai_override_log_df = pd.DataFrame()
    ai_impact_summary = {}

    if enable_ai_review and api_key_present and not combined_bulk_updates.empty:
        try:
            ai_candidates_df = build_ai_action_candidates(
                combined_bulk_updates=combined_bulk_updates,
                bid_recommendations=bid_recommendations,
                search_term_actions=search_term_actions,
                campaign_budget_actions=campaign_budget_actions,
            )

            low_conf_df = ai_candidates_df[ai_candidates_df["confidence"].isin(["LOW", "MEDIUM"])].copy()

            if not low_conf_df.empty:
                payload = build_ai_override_payload(
                    low_conf_df=low_conf_df,
                    account_summary=output_account_summary,
                    account_health=output_account_health,
                    simulation_summary=simulation_summary,
                    sqp_opportunities=output_sqp_opportunities,
                    sqp_summary=output_sqp_summary,
                )

                ai_response = run_ai_review_cached(json.dumps(payload))
                original_count = len(combined_bulk_updates)

                combined_bulk_updates, ai_override_log_df = apply_ai_overrides_to_combined(
                    combined_bulk_updates=combined_bulk_updates,
                    ai_response=ai_response,
                    ai_candidates_df=ai_candidates_df,
                )

                final_count = len(combined_bulk_updates)
                ai_impact_summary = build_ai_impact_summary(
                    ai_override_log_df=ai_override_log_df,
                    original_count=original_count,
                    final_count=final_count,
                    executive_summary=ai_response.get("executive_summary", ""),
                )
        except Exception as e:
            st.warning(f"AI optimization skipped: {e}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Optimization Summary</div>', unsafe_allow_html=True)
    
        preview_summary = safe_dict(outputs.get("pre_run_preview"))
        found_graduations = int(preview_summary.get("graduations", 0))
        ready_graduations = int(simulation_summary.get("graduations", 0))
    
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        with s1:
            render_metric_card("Bid Increases", str(simulation_summary.get("bid_increases", 0)), "good")
        with s2:
            render_metric_card("Bid Decreases", str(simulation_summary.get("bid_decreases", 0)), "warn")
        with s3:
            render_metric_card("Neg. Exacts", str(simulation_summary.get("negatives_added", 0)), "warn")
        with s4:
            render_metric_card("Grad Opportunities", str(found_graduations), "good")
        with s5:
            render_metric_card("Grad Ready This Upload", str(ready_graduations), "brand")
        with s6:
            render_metric_card(
                "Budget Actions",
                str(simulation_summary.get("budget_increases", 0) + simulation_summary.get("budget_decreases", 0)),
                "brand",
            )
    
        st.markdown("### Breakdown")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric(
                "Dest Exact",
                preview_summary.get("dest_exact", 0),
                delta=f"Ready now: {simulation_summary.get('dest_exact', 0)}",
            )
        with b2:
            st.metric(
                "Research Phrase",
                preview_summary.get("research_phrase", 0),
                delta=f"Ready now: {simulation_summary.get('research_phrase', 0)}",
            )
        with b3:
            st.metric(
                "ASIN Dest",
                preview_summary.get("asin_dest", 0),
                delta=f"Ready now: {simulation_summary.get('asin_dest', 0)}",
            )
        with b4:
            st.metric("Ad Group Creates", simulation_summary.get("ad_group_creates", 0))


            if found_graduations > ready_graduations:
                st.info(
                    f"The engine found {found_graduations} graduation opportunities, but only {ready_graduations} "
                    f"were ready for this upload. The remaining opportunities need existing destination campaign/ad group IDs "
                    f"before Amazon will accept the child rows."
                )

    narrative_notes = build_narrative(
        account_health=output_account_health,
        simulation_summary=simulation_summary,
        enable_monthly_budget_control=enable_monthly_budget_control,
        pacing_status=pacing_status,
    )
    st.markdown("### Optimization Narrative")
    for note in narrative_notes:
        st.markdown(f"- {note}")

    if ai_impact_summary:
        st.markdown("### AI Review Summary")
        st.markdown(f"- Reviewed: {ai_impact_summary.get('reviewed', 0)}")
        st.markdown(f"- Modified: {ai_impact_summary.get('modified', 0)}")
        st.markdown(f"- Removed: {ai_impact_summary.get('removed', 0)}")
        if ai_impact_summary.get("executive_summary"):
            st.markdown(f"- {ai_impact_summary['executive_summary']}")

    tabs = st.tabs([
        "Amazon Bulk Upload",
        "Dest Structure Builder",
        "Deferred Graduations",
        "Bid Recommendations",
        "Search Term Actions",
        "Budget Actions",
        "Campaign Health",
        "SQP Opportunities",
        "AI Override Log",
        "Run History",
    ])

    with tabs[0]:
        if not combined_bulk_updates.empty:
            st.dataframe(combined_bulk_updates, use_container_width=True)
        else:
            st.info("No bulk upload output available.")

    with tabs[1]:
        if not dest_structure_upload.empty:
            st.dataframe(dest_structure_upload, use_container_width=True)
        else:
            st.info("No missing Dest / Research structure needs to be created.")
    
    with tabs[2]:
        if not deferred_graduations.empty:
            st.dataframe(deferred_graduations, use_container_width=True)
        else:
            st.info("No deferred graduations for this run.")        

    with tabs[3]:
        if not bid_recommendations.empty:
            st.dataframe(bid_recommendations, use_container_width=True)
        else:
            st.info("No bid recommendations available.")

    with tabs[4]:
        if not search_term_actions.empty:
            st.dataframe(search_term_actions, use_container_width=True)
        else:
            st.info("No search term actions available.")

    with tabs[5]:
        if not campaign_budget_actions.empty:
            st.dataframe(campaign_budget_actions, use_container_width=True)
        else:
            st.info("No campaign budget actions available.")

    with tabs[6]:
        if not output_campaign_health_dashboard.empty:
            st.dataframe(output_campaign_health_dashboard, use_container_width=True)
        else:
            st.info("No campaign health table available.")

    with tabs[7]:
        if not output_sqp_opportunities.empty:
            st.dataframe(output_sqp_opportunities, use_container_width=True)
        else:
            st.info("No SQP opportunities available.")

    with tabs[8]:
        if not ai_override_log_df.empty:
            st.dataframe(ai_override_log_df, use_container_width=True)
        else:
            st.info("No AI override log available for this run.")

    with tabs[9]:
        if not run_history.empty:
            st.dataframe(run_history, use_container_width=True)
        else:
            st.info("No run history available.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Downloads</div>', unsafe_allow_html=True)

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    with d1:
        st.download_button(
            label="Download Amazon Bulk Upload",
            data=to_excel_bytes(combined_bulk_updates),
            file_name="amazon_bulk_updates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="Download Bid Recommendations",
            data=to_excel_bytes(bid_recommendations),
            file_name="bid_recommendations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            label="Download Search Term Actions",
            data=to_excel_bytes(search_term_actions),
            file_name="search_term_actions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with d4:
        st.download_button(
            label="Download Campaign Budget Actions",
            data=to_excel_bytes(campaign_budget_actions),
            file_name="campaign_budget_actions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with d5:
        st.download_button(
            label="Download Dest Structure Builder",
            data=to_excel_bytes(dest_structure_upload),
            file_name="dest_structure_builder.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    
    with d6:
        st.download_button(
            label="Download Deferred Graduations",
            data=to_excel_bytes(deferred_graduations),
            file_name="deferred_graduations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )    
