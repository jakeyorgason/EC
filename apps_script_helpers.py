import requests
import streamlit as st


def create_google_sheet_report(
    brand_name: str,
    report_name: str,
    date_range_label: str,
    kpi_summary: dict,
    waste_summary: dict,
    match_type_revenue_rows: list[dict],
    match_type_inefficient_rows: list[dict],
):
    webhook_url = st.secrets["APPS_SCRIPT_WEBHOOK_URL"]
    template_id = st.secrets["GOOGLE_SHEETS_TEMPLATE_ID"]
    destination_folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]

    payload = {
        "templateId": template_id,
        "destinationFolderId": destination_folder_id,
        "reportName": report_name,
        "brandName": brand_name,
        "dateRangeLabel": date_range_label,
        "kpiSummary": kpi_summary,
        "wasteSummary": waste_summary,
        "matchTypeRevenueRows": match_type_revenue_rows,
        "matchTypeInefficientRows": match_type_inefficient_rows,
    }

    response = requests.post(webhook_url, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()
    if not data.get("success"):
        raise RuntimeError(data.get("error", "Unknown Apps Script error"))

    return data
