import io
import os
import calendar
from datetime import date
from typing import Optional, Any

import pandas as pd
import streamlit as st
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

        .metric-value {
            font-size: 1.5rem;   /* ↓ from 2rem */
            line-height: 1.1;
            font-weight: 800;
            color: #1F2937;
            word-break: break-word;
        }

        .metric-value.small {
            font-size: 1.15rem;  /* ↓ from 1.35rem */
        }

        .metric-label {
            font-size: 0.75rem;
            color: #6B7280;
            margin-bottom: 0.25rem;
            font-weight: 600;
        }

        .metric-card {
            padding: 12px 14px;   /* slightly tighter */
            min-height: 80px;     /* ↓ from 92px */
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

    st.markdown("## Optional upload")
    st.markdown("- Seller Central Business Report for TACOS control")

    st.markdown("---")
    st.markdown("## Recommended Workflow")
    st.markdown(
        """
- Start with **Balanced**
- Use **Losing KW Action = Both**
- Use monthly pacing only for budget-sensitive clients
- Review diagnostics before running
- Review output tables before download
"""
    )

    st.markdown("---")
    st.markdown("**Version:** 1.1")
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
                Bid optimization • Search harvesting • Negative mining • Budget pacing • TACOS control
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Settings
# =========================================================
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
        "Minimum Clicks Before Efficiency Actions",
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
    '<div class="section-note">Upload the report set to unlock diagnostics and optimization preview.</div>',
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

bulk_bytes = get_uploaded_bytes(bulk_file)
search_bytes = get_uploaded_bytes(search_file)
targeting_bytes = get_uploaded_bytes(targeting_file)
impression_bytes = get_uploaded_bytes(impression_file)
business_bytes = get_uploaded_bytes(business_file)

required_ready = all([bulk_bytes, search_bytes, targeting_bytes, impression_bytes])
tacos_ready = (not enable_tacos_control) or (business_bytes is not None)


def build_engine() -> AdsOptimizerEngine:
    return AdsOptimizerEngine(
        bulk_file=bytes_to_buffer(bulk_bytes),
        search_term_file=bytes_to_buffer(search_bytes),
        targeting_file=bytes_to_buffer(targeting_bytes),
        impression_share_file=bytes_to_buffer(impression_bytes),
        business_report_file=bytes_to_buffer(business_bytes),
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
        render_metric_card(
            "Ad Spend",
            f"${get_number(account_summary.get('total_spend')):,.2f}",
            tone="brand",
        )

    with dh2:
        render_metric_card(
            "Ad Sales",
            f"${get_number(account_summary.get('total_sales')):,.2f}",
            tone="brand",
        )

    with dh3:
        roas_tone = "good" if account_roas >= adjusted_min_roas else "bad"
        render_metric_card("Account ROAS", f"{account_roas:.2f}", tone=roas_tone)

    with dh4:
        render_metric_card("TACOS", str(tacos_display), tone="warn", small=True)

    with dh5:
        render_metric_card(
            "Under Target",
            str(get_int(account_summary.get("campaigns_under_target"))),
            tone="bad",
        )

    with dh6:
        render_metric_card(
            "Scalable",
            str(get_int(account_summary.get("campaigns_scalable"))),
            tone="good",
        )

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

if enable_tacos_control:
    st.markdown(
        f"- If account TACOS rises above **{max_tacos_target:.1f}%**, the app will tighten scaling rules."
    )

if enable_monthly_budget_control and monthly_account_budget > 0:
    st.markdown(
        "- Monthly budget control is enabled. If current pacing exceeds the allowed pace for the month, the app will block budget increases and block bid increases."
    )


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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Optimization Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">High-level actions generated by this run.</div>',
        unsafe_allow_html=True,
    )

    summary1, summary2, summary3, summary4, summary5, summary6 = st.columns(6)

    with summary1:
        render_metric_card(
            "Bid Increases",
            str(get_int(simulation_summary.get("bid_increases"))),
            tone="good",
        )

    with summary2:
        render_metric_card(
            "Bid Decreases",
            str(get_int(simulation_summary.get("bid_decreases"))),
            tone="warn",
        )

    with summary3:
        render_metric_card(
            "Negatives",
            str(get_int(simulation_summary.get("negatives_added"))),
            tone="warn",
        )

    with summary4:
        render_metric_card(
            "Harvests",
            str(get_int(simulation_summary.get("harvested_keywords"))),
            tone="good",
        )

    with summary5:
        render_metric_card(
            "Budget Increases",
            str(get_int(simulation_summary.get("budget_increases"))),
            tone="good",
        )

    with summary6:
        render_metric_card(
            "Budget Decreases",
            str(get_int(simulation_summary.get("budget_decreases"))),
            tone="warn",
        )

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

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Bulk Upload",
            "Bid Recommendations",
            "Search Term Actions",
            "Budget Actions",
        ]
    )

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