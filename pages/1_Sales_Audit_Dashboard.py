import os
from typing import Any, Optional

import pandas as pd
import streamlit as st

from sales_audit_ingestion import SalesAuditEngine
from shared_ingestion_utils import to_excel_bytes_multi

from apps_script_helpers import create_google_sheet_report

st.set_page_config(
    page_title="Sales Audit | Amazon Ads Command Center",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# HELPERS
# =========================================================
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


def format_currency(value: float) -> str:
    return f"${get_number(value):,.2f}"


def format_percent(value: float) -> str:
    return f"{get_number(value):,.2f}%"


def format_number(value: float) -> str:
    return f"{get_number(value):,.2f}"


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


def tone_from_health(status: str) -> str:
    status = str(status).strip().lower()
    if status == "healthy":
        return "good"
    if status == "mixed":
        return "warn"
    return "bad"


def simplify_term_table(df: pd.DataFrame, term_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "acos" in out.columns and "acos_pct" not in out.columns:
        out["acos_pct"] = out["acos"] * 100

    keep_cols = [c for c in [term_col, "spend", "sales", "acos_pct"] if c in out.columns]
    out = out[keep_cols].copy()

    rename_map = {
        term_col: "term",
        "spend": "spend",
        "sales": "sales",
        "acos_pct": "acos",
    }
    out = out.rename(columns=rename_map)

    out["spend"] = pd.to_numeric(out["spend"], errors="coerce").fillna(0)
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce").fillna(0)
    out["acos"] = pd.to_numeric(out["acos"], errors="coerce").fillna(0)

    out["spend"] = out["spend"].map(lambda x: f"${x:,.2f}")
    out["sales"] = out["sales"].map(lambda x: f"${x:,.2f}")
    out["acos"] = out["acos"].map(lambda x: f"{x:,.2f}%")

    return out.reset_index(drop=True)


def simplify_campaign_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    keep = [c for c in ["campaign_name", "spend", "sales", "acos_pct", "campaign_status"] if c in out.columns]
    out = out[keep].copy()

    if "spend" in out.columns:
        out["spend"] = pd.to_numeric(out["spend"], errors="coerce").fillna(0).round(2)
    if "sales" in out.columns:
        out["sales"] = pd.to_numeric(out["sales"], errors="coerce").fillna(0).round(2)
    if "acos_pct" in out.columns:
        out["acos_pct"] = pd.to_numeric(out["acos_pct"], errors="coerce").fillna(0).round(2)

    return out.sort_values(["spend", "sales"], ascending=[False, False]).reset_index(drop=True)

def df_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []

    out = df.copy()
    out = out.replace({pd.NA: None})
    out = out.where(pd.notnull(out), None)
    return out.to_dict("records")    


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1.75rem;
            max-width: 1440px;
        }
        
        .brand-shell {
            background: linear-gradient(135deg, #EA580C 0%, #1F2937 100%);
            border-radius: 20px;
            padding: 65px 30px;
            color: white;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 05px rgba(0, 0, 0, 0.12);
        }

        .brand-title {
            font-size: 2.15rem;
            line-height: 1.05;
            font-weight: 800;
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
            font-size: 0.94rem;
            color: #4B5563;
            margin-bottom: 0.9rem;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #ececec;
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            min-height: 98px;
        }

        .metric-label {
            font-size: 0.75rem;
            color: #6B7280;
            margin-bottom: 0.25rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-value {
            font-size: 1.5rem;
            line-height: 1.1;
            font-weight: 800;
            color: #1F2937;
            word-break: break-word;
        }

        .metric-value.small {
            font-size: 1.1rem;
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

        .status-pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }

        .status-pill.good {
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
        }

        .status-pill.warn {
            background: #fffbeb;
            color: #b45309;
            border: 1px solid #fde68a;
        }

        .status-pill.bad {
            background: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }

        .summary-box {
            background: #ffffff;
            border: 1px solid #ececec;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 10px;
        }

        .upload-note {
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: -0.1rem;
            margin-bottom: 0.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Sales Audit Guide")
    st.markdown(
        """
**Purpose**
- Give sales a quick account health snapshot
- Surface wasted spend
- Identify major spend drivers
- Flag inefficient terms above threshold
- Highlight winner terms with low ACOS
"""
    )

    st.markdown("---")
    st.markdown("## Required uploads")
    st.markdown(
        """
1. Bulk Sheet  
2. Impression Share Report  
3. Targeting Report  
4. Search Term Report  
5. Sales & Traffic Business Report
"""
    )

    st.markdown("---")
    st.markdown("## Recommended use")
    st.markdown(
        """
- Use recent reporting windows across all files
- Keep thresholds consistent across audits
- Focus on prospect-facing takeaways
- Use the workbook export as backup detail
"""
    )

    st.markdown("---")
    st.markdown("**Version:** 1.1")
    st.markdown("**Owner:** Jake Yorgason, Evolved Commerce")


# =========================================================
# HEADER
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
            <div class="brand-title">Evolved Commerce<br>Sales Audit Dashboard</div>
            <div class="brand-subtitle">
                Ads Health • Effeciency Diagnosis • Waste Identification • Spend Allocation • Keyword Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
brand_name = st.text_input(
    "Brand Name",
    placeholder="Brand Name",
    key="sales_audit_brand_name",
)

# =========================================================
# SETTINGS
# =========================================================
st.markdown('<div class="section-title">Audit Settings</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">These thresholds control waste flags, winner logic, and the audit verdict.</div>',
    unsafe_allow_html=True,
)

s1, s2, s3, s4 = st.columns(4)

with s1:
    high_acos_threshold = st.number_input(
        "High ACOS Threshold %",
        min_value=1.0,
        max_value=100.0,
        value=40.0,
        step=1.0,
    )

with s2:
    winning_acos_threshold = st.number_input(
        "Winner ACOS Threshold %",
        min_value=1.0,
        max_value=100.0,
        value=25.0,
        step=1.0,
    )

with s3:
    min_waste_spend = st.number_input(
        "Minimum Spend to Flag Waste",
        min_value=0.0,
        max_value=1000.0,
        value=20.0,
        step=5.0,
    )

with s4:
    min_winner_orders = st.number_input(
        "Minimum Orders for Winner",
        min_value=1,
        max_value=50,
        value=2,
        step=1,
    )


# =========================================================
# UPLOADS
# =========================================================
st.markdown('<div class="section-title">Required Uploads</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Upload all five files before running the audit.</div>',
    unsafe_allow_html=True,
)

u1, u2 = st.columns(2)
u3, u4 = st.columns(2)
u5, _ = st.columns([1, 1])

with u1:
    bulk_file = st.file_uploader(
        "Bulk Sheet",
        type=["xlsx", "xls", "csv"],
        key="sales_audit_bulk_file",
    )

with u2:
    impression_share_file = st.file_uploader(
        "Impression Share Report",
        type=["csv", "xlsx", "xls"],
        key="sales_audit_impression_share_file",
    )

with u3:
    targeting_file = st.file_uploader(
        "Targeting Report",
        type=["csv", "xlsx", "xls"],
        key="sales_audit_targeting_file",
    )

with u4:
    search_term_file = st.file_uploader(
        "Search Term Report",
        type=["csv", "xlsx", "xls"],
        key="sales_audit_search_term_file",
    )

with u5:
    business_report_file = st.file_uploader(
        "Sales & Traffic Business Report",
        type=["csv", "xlsx", "xls"],
        key="sales_audit_business_report_file",
    )

required_ready = all([
    bulk_file is not None,
    impression_share_file is not None,
    targeting_file is not None,
    search_term_file is not None,
    business_report_file is not None,
])

st.markdown(
    f'<div class="upload-note">Status: {"Ready to run" if required_ready else "Waiting on all 5 required uploads"}</div>',
    unsafe_allow_html=True,
)

run_clicked = st.button(
    "Run Sales Audit",
    type="primary",
    use_container_width=True,
    disabled=not required_ready,
)


# =========================================================
# PROCESS
# =========================================================
if run_clicked:
    st.session_state.pop("sales_audit_results", None)

    with st.spinner("Running sales audit..."):
        try:
            engine = SalesAuditEngine(
                bulk_file=bulk_file,
                impression_share_file=impression_share_file,
                targeting_file=targeting_file,
                search_term_file=search_term_file,
                business_report_file=business_report_file,
                high_acos_threshold=high_acos_threshold,
                winning_acos_threshold=winning_acos_threshold,
                min_waste_spend=min_waste_spend,
                min_winner_orders=min_winner_orders,
            )

            results = engine.process()
            results["brand_name"] = (brand_name or "").strip()
            st.session_state["sales_audit_results"] = results
            st.success("Sales audit complete.")

        except Exception as exc:
            st.error(f"Sales audit failed: {exc}")

results = safe_dict(st.session_state.get("sales_audit_results", {}))
brand_name = str(results.get("brand_name", "")).strip()

if results:
    kpis = safe_dict(results.get("kpi_summary"))
    waste_summary = safe_dict(results.get("waste_summary"))
    health_summary = safe_dict(results.get("health_summary"))
    campaign_summary = safe_df(results.get("campaign_summary"))
    keyword_spend_table = safe_df(results.get("keyword_spend_table"))
    search_term_spend_table = safe_df(results.get("search_term_spend_table"))
    waste_tables = safe_dict(results.get("waste_tables"))
    winner_tables = safe_dict(results.get("winner_tables"))
    narrative = str(results.get("narrative", "")).strip()

    with st.expander("Match Type Source Debug", expanded=False):
        search_terms_df = safe_df(results.get("search_terms"))
        if not search_terms_df.empty and "match_type" in search_terms_df.columns:
            st.write("Unique search-term match types:", sorted(search_terms_df["match_type"].dropna().astype(str).unique().tolist()))
            st.write(search_terms_df[["match_type", "impressions", "clicks", "spend", "sales"]].head(20))
        else:
            st.write("No search_terms match_type column found.")

    kw_zero = simplify_term_table(safe_df(waste_tables.get("keyword_zero_sale")), "target")
    kw_high = simplify_term_table(safe_df(waste_tables.get("keyword_high_acos")), "target")
    st_zero = simplify_term_table(safe_df(waste_tables.get("search_zero_sale")), "customer_search_term")
    st_high = simplify_term_table(safe_df(waste_tables.get("search_high_acos")), "customer_search_term")

    kw_winners = simplify_term_table(safe_df(winner_tables.get("keyword_winners")), "target").head(20)
    st_winners = simplify_term_table(safe_df(winner_tables.get("search_winners")), "customer_search_term").head(20)

    top_kw = simplify_term_table(keyword_spend_table, "target").head(20)
    top_st = simplify_term_table(search_term_spend_table, "customer_search_term").head(20)
    campaign_view = simplify_campaign_table(campaign_summary).head(20)

    st.markdown('<div class="section-title">Create Branded Google Sheet Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Creates a branded Google Sheets audit report in your Sales Audits folder.</div>',
    unsafe_allow_html=True,)

    report_name_default = f"{brand_name} - Sales Audit" if brand_name else "Sales Audit Report"
    report_name = st.text_input(
        "Google Sheet Report Name",
        value=report_name_default,
        key="sales_audit_google_sheet_report_name",
    )

    if st.button("Create Branded Google Sheet Report", use_container_width=True):
        try:
            if not brand_name:
                st.error("Please enter a Brand Name before creating the Google Sheet report.")
            else:
                waste_kw_combined = pd.concat([kw_zero, kw_high], ignore_index=True).drop_duplicates()
                winner_combined = pd.concat([kw_winners, st_winners], ignore_index=True).drop_duplicates()

                date_range_label = "MM/DD - MM/DD"

                created_report = create_google_sheet_report(
                    brand_name=brand_name,
                    report_name=report_name,
                    date_range_label=date_range_label,
                    kpi_summary=kpis,
                    waste_summary=waste_summary,
                    match_type_revenue_rows=results.get("match_type_revenue_rows", []),
                    match_type_inefficient_rows=results.get("match_type_inefficient_rows", []),
                )

                st.success("Branded Google Sheet report created successfully.")
                st.markdown(f"[Open Google Sheet]({created_report['url']})")
        except Exception as exc:
            st.error(f"Google Sheet report creation failed: {exc}")

    # =========================================================
    # HEALTH SUMMARY
    # =========================================================
    if brand_name:
        st.markdown(f"### {brand_name} Sales Audit")

    st.markdown('<div class="section-title">Account Health Verdict</div>', unsafe_allow_html=True)

    status = health_summary.get("status", "Unknown")
    tone = tone_from_health(status)

    st.markdown(
        f'<div class="status-pill {tone}">{status}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="summary-box">
            <strong>Summary:</strong> {health_summary.get("summary", "No summary available.")}
            <br><br>
            <strong>Narrative:</strong> {narrative}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # KPI CARDS
    # =========================================================
    st.markdown('<div class="section-title">Executive KPI Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Top-level account performance, blending ad efficiency with total business sales.</div>',
        unsafe_allow_html=True,
    )

    r1 = st.columns(4)
    with r1[0]:
        render_metric_card("Spend", format_currency(kpis.get("spend")), tone="brand")
    with r1[1]:
        render_metric_card("Ad Sales", format_currency(kpis.get("ad_sales")), tone="brand")
    with r1[2]:
        render_metric_card("Total Sales", format_currency(kpis.get("total_sales")), tone="brand")
    with r1[3]:
        render_metric_card("Organic Sales", format_currency(kpis.get("organic_sales")), tone="brand")

    r2 = st.columns(4)
    with r2[0]:
        render_metric_card("ACOS", format_percent(kpis.get("acos_pct")), tone="warn")
    with r2[1]:
        render_metric_card("ROAS", format_number(kpis.get("roas")), tone="good")
    with r2[2]:
        render_metric_card("TACOS", format_percent(kpis.get("tacos_pct")), tone="warn")
    with r2[3]:
        render_metric_card("Organic Share", format_percent(kpis.get("organic_share_pct")), tone="good")

    r3 = st.columns(3)
    with r3[0]:
        render_metric_card("Wasted Spend", format_currency(waste_summary.get("wasted_spend")), tone="bad")
    with r3[1]:
        render_metric_card("Wasted Spend %", format_percent(waste_summary.get("wasted_spend_pct")), tone="bad")
    with r3[2]:
        render_metric_card(
            "Post-Ad Contribution",
            format_currency(kpis.get("estimated_post_ad_contribution")),
            tone="brand",
            small=True,
        )

    # =========================================================
    # TOP SPENDERS
    # =========================================================
    st.markdown('<div class="section-title">Top Spenders</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Prospect-facing view of the biggest spend drivers.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Top Keyword / Target Spenders**")
        if not top_kw.empty:
            st.dataframe(top_kw, use_container_width=True)
        else:
            st.info("No keyword/target spend table available.")

    with c2:
        st.markdown("**Top Customer Search Term Spenders**")
        if not top_st.empty:
            st.dataframe(top_st, use_container_width=True)
        else:
            st.info("No customer search term spend table available.")

    # =========================================================
    # WASTE
    # =========================================================
    st.markdown('<div class="section-title">Waste Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Terms with zero sales or ACOS above the threshold.</div>',
        unsafe_allow_html=True,
    )

    waste_left, waste_right = st.columns(2)

    with waste_left:
        st.markdown("**Keyword / Target Waste**")
        waste_kw_combined = pd.concat([kw_zero, kw_high], ignore_index=True).drop_duplicates().head(20)
        if not waste_kw_combined.empty:
            st.dataframe(waste_kw_combined, use_container_width=True)
        else:
            st.info("No keyword/target waste found.")

    with waste_right:
        st.markdown("**Customer Search Term Waste**")
        waste_st_combined = pd.concat([st_zero, st_high], ignore_index=True).drop_duplicates().head(20)
        if not waste_st_combined.empty:
            st.dataframe(waste_st_combined, use_container_width=True)
        else:
            st.info("No customer search term waste found.")

    # =========================================================
    # WINNERS
    # =========================================================
    st.markdown('<div class="section-title">Winning Terms</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Terms meeting the order floor and staying at or below the winner ACOS threshold.</div>',
        unsafe_allow_html=True,
    )

    w1, w2 = st.columns(2)

    with w1:
        st.markdown("**Winning Keyword / Targets**")
        if not kw_winners.empty:
            st.dataframe(kw_winners, use_container_width=True)
        else:
            st.info("No winning keyword targets found.")

    with w2:
        st.markdown("**Winning Customer Search Terms**")
        if not st_winners.empty:
            st.dataframe(st_winners, use_container_width=True)
        else:
            st.info("No winning customer search terms found.")

    # =========================================================
    # OPTIONAL CAMPAIGN VIEW
    # =========================================================
    with st.expander("Campaign Summary", expanded=False):
        if not campaign_view.empty:
            st.dataframe(campaign_view, use_container_width=True)
        else:
            st.info("No campaign summary available.")

    # =========================================================
    # EXPORTS
    # =========================================================
    st.markdown('<div class="section-title">Export Audit Workbook</div>', unsafe_allow_html=True)

    export_sheets = {
        "KPI Summary": pd.DataFrame([kpis]),
        "Waste Summary": pd.DataFrame([waste_summary]),
        "Campaign Summary": campaign_view,
        "Keyword Spend": simplify_term_table(keyword_spend_table, "target"),
        "Search Term Spend": simplify_term_table(search_term_spend_table, "customer_search_term"),
        "KW Waste": pd.concat([kw_zero, kw_high], ignore_index=True).drop_duplicates(),
        "ST Waste": pd.concat([st_zero, st_high], ignore_index=True).drop_duplicates(),
        "KW Winners": kw_winners,
        "ST Winners": st_winners,
    }

    export_bytes = to_excel_bytes_multi(export_sheets)

    st.download_button(
        label="Download Sales Audit Workbook",
        data=export_bytes,
        file_name=f"{brand_name or 'sales_audit'}_audit_workbook.xlsx".replace(" ", "_"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    st.info("Upload the five required files, then click Run Sales Audit.")
