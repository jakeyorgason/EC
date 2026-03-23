import streamlit as st
import gspread

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def get_google_service_account_info():
    return {
        "type": st.secrets["GOOGLE_TYPE"],
        "project_id": st.secrets["GOOGLE_PROJECT_ID"],
        "private_key_id": st.secrets["GOOGLE_PRIVATE_KEY_ID"],
        "private_key": st.secrets["GOOGLE_PRIVATE_KEY"],
        "client_email": st.secrets["GOOGLE_CLIENT_EMAIL"],
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "auth_uri": st.secrets["GOOGLE_AUTH_URI"],
        "token_uri": st.secrets["GOOGLE_TOKEN_URI"],
        "auth_provider_x509_cert_url": st.secrets["GOOGLE_AUTH_PROVIDER_X509_CERT_URL"],
        "client_x509_cert_url": st.secrets["GOOGLE_CLIENT_X509_CERT_URL"],
    }


def get_google_credentials():
    service_account_info = get_google_service_account_info()
    return Credentials.from_service_account_info(
        service_account_info,
        scopes=GOOGLE_SCOPES,
    )


def get_gspread_client():
    creds = get_google_credentials()
    return gspread.authorize(creds)


def get_drive_service():
    creds = get_google_credentials()
    return build("drive", "v3", credentials=creds)


def get_audit_folder_id():
    return st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
