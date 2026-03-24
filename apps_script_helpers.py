import requests
import streamlit as st


def create_google_sheet_from_template(report_name: str):
    webhook_url = st.secrets["APPS_SCRIPT_WEBHOOK_URL"]
    template_id = st.secrets["GOOGLE_SHEETS_TEMPLATE_ID"]
    destination_folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]

    payload = {
        "templateId": template_id,
        "destinationFolderId": destination_folder_id,
        "reportName": report_name,
    }

    response = requests.post(webhook_url, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    if not data.get("success"):
        raise RuntimeError(data.get("error", "Unknown Apps Script error"))

    return data
