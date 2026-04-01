import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class CampaignMeta:
    campaign_name: str
    campaign_id: str = ""
    portfolio_name: str = ""
    portfolio_id: str = ""
    advertised_item: str = ""
    campaign_type: str = ""
    description: str = ""
    suffix: str = ""
    entity: str = "Campaign"


class AdsOptimizerEngine:
    """Rule-based Amazon Ads optimizer with keyword graduation support.

    This version keeps the original app surface area, but adds a richer search-term
    routing engine that can:
    - graduate exact winners into destination campaigns
    - route weaker winners into research phrase campaigns
    - route ASIN winners into an ASIN Targets ad group inside destination campaigns
    - automatically create missing destination / research campaigns and ad groups
    - always negate the source ad group after graduation
    """

    def __init__(
        self,
        bulk_file,
        search_term_file,
        targeting_file,
        impression_share_file,
        business_report_file=None,
        sqp_report_file=None,
        min_roas=3.0,
        min_clicks=8,
        zero_order_click_threshold=12,
        zero_order_action="Both",
        strategy_mode="Balanced",
        enable_bid_updates=True,
        enable_search_harvesting=True,
        enable_negative_keywords=True,
        enable_budget_updates=True,
        enable_tacos_control=False,
        max_tacos_target=15.0,
        enable_monthly_budget_control=False,
        monthly_account_budget=0.0,
        month_to_date_spend=0.0,
        pacing_buffer_pct=5.0,
        max_bid_cap=5.00,
        max_budget_cap=500.00,
        graduation_orders_threshold=2,
        dest_acos_threshold_pct=25.0,
        loser_clicks_threshold=5,
        loser_ctr_threshold_pct=0.25,
        loser_cvr_threshold_pct=5.0,
        research_ctr_min_pct=0.10,
        research_ctr_max_pct=0.25,
        research_cvr_min_pct=2.0,
        research_cvr_max_pct=5.0,
        new_target_bid_multiplier=1.10,
        new_target_bid_cap=1.00,
    ):
        self.bulk_file = bulk_file
        self.search_term_file = search_term_file
        self.targeting_file = targeting_file
        self.impression_share_file = impression_share_file
        self.business_report_file = business_report_file
        self.sqp_report_file = sqp_report_file

        self.min_roas = float(min_roas)
        self.min_clicks = int(min_clicks)
        self.zero_order_click_threshold = int(zero_order_click_threshold)
        self.zero_order_action = zero_order_action
        self.strategy_mode = strategy_mode

        self.enable_bid_updates = enable_bid_updates
        self.enable_search_harvesting = enable_search_harvesting
        self.enable_negative_keywords = enable_negative_keywords
        self.enable_budget_updates = enable_budget_updates
        self.enable_tacos_control = enable_tacos_control
        self.max_tacos_target = float(max_tacos_target) / 100.0
        self.enable_monthly_budget_control = enable_monthly_budget_control
        self.monthly_account_budget = float(monthly_account_budget)
        self.month_to_date_spend = float(month_to_date_spend)
        self.pacing_buffer_pct = float(pacing_buffer_pct) / 100.0
        self.max_bid_cap = float(max_bid_cap)
        self.max_budget_cap = float(max_budget_cap)

        self.graduation_orders_threshold = int(graduation_orders_threshold)
        self.dest_acos_threshold = float(dest_acos_threshold_pct) / 100.0
        self.loser_clicks_threshold = int(loser_clicks_threshold)
        self.loser_ctr_threshold = float(loser_ctr_threshold_pct) / 100.0
        self.loser_cvr_threshold = float(loser_cvr_threshold_pct) / 100.0
        self.research_ctr_min = float(research_ctr_min_pct) / 100.0
        self.research_ctr_max = float(research_ctr_max_pct) / 100.0
        self.research_cvr_min = float(research_cvr_min_pct) / 100.0
        self.research_cvr_max = float(research_cvr_max_pct) / 100.0
        self.new_target_bid_multiplier = float(new_target_bid_multiplier)
        self.new_target_bid_cap = float(new_target_bid_cap)

        self.run_history_path = Path("/mnt/data/ads_optimizer_run_history.json")

        self.bulk_columns: list[str] = []
        self.bulk_campaign_inventory = pd.DataFrame()
        self.bulk_ad_group_inventory = pd.DataFrame()
        self.bulk_target_inventory = pd.DataFrame()
        self.dest_campaign_inventory = pd.DataFrame()
        self.research_campaign_inventory = pd.DataFrame()

        self.existing_exact_keywords = set()
        self.existing_phrase_keywords = set()
        self.existing_product_targets = set()
        self.existing_negative_exact = set()
        self.existing_ad_groups = set()

        self.apply_strategy_settings()

    # -----------------------------
    # STRATEGY SETTINGS
    # -----------------------------
    def apply_strategy_settings(self):
        mode = str(self.strategy_mode).strip().lower()
        if mode == "conservative":
            self.max_bid_up = 0.05
            self.max_bid_down = 0.10
            self.scale_roas_multiplier = 1.40
            self.budget_up_pct = 0.05
            self.budget_down_pct = 0.08
            self.min_orders_for_scaling = 3
            self.account_health_tighten_multiplier = 1.10
        elif mode == "aggressive":
            self.max_bid_up = 0.20
            self.max_bid_down = 0.25
            self.scale_roas_multiplier = 1.10
            self.budget_up_pct = 0.15
            self.budget_down_pct = 0.12
            self.min_orders_for_scaling = 2
            self.account_health_tighten_multiplier = 0.90
        else:
            self.max_bid_up = 0.10
            self.max_bid_down = 0.15
            self.scale_roas_multiplier = 1.25
            self.budget_up_pct = 0.10
            self.budget_down_pct = 0.10
            self.min_orders_for_scaling = 2
            self.account_health_tighten_multiplier = 1.00

    # -----------------------------
    # FILE HELPERS
    # -----------------------------
    def _clone_file_obj(self, file_obj):
        if file_obj is None:
            return None
        if isinstance(file_obj, (str, bytes, os.PathLike)):
            return file_obj
        if isinstance(file_obj, io.BytesIO):
            return io.BytesIO(file_obj.getvalue())
        if hasattr(file_obj, "getvalue"):
            return io.BytesIO(file_obj.getvalue())
        if hasattr(file_obj, "read"):
            cur = None
            try:
                cur = file_obj.tell()
            except Exception:
                pass
            try:
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                data = file_obj.read()
            finally:
                try:
                    if cur is not None and hasattr(file_obj, "seek"):
                        file_obj.seek(cur)
                except Exception:
                    pass
            return io.BytesIO(data)
        raise ValueError("Unsupported file object type")

    def _get_file_extension(self, file_obj, fallback=None):
        if file_obj is None:
            return None
        if isinstance(file_obj, (str, os.PathLike)):
            return os.path.splitext(str(file_obj))[1].lower()
        name = getattr(file_obj, "name", None)
        if isinstance(name, str) and name.strip():
            return os.path.splitext(name)[1].lower()
        return fallback

    def load_file(self, file_obj, expected_ext=None):
        if file_obj is None:
            return None
        ext = self._get_file_extension(file_obj, fallback=expected_ext)
        cloned = self._clone_file_obj(file_obj)
        if ext == ".csv":
            return pd.read_csv(cloned)
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(cloned, engine="openpyxl")
        raise ValueError(f"Unsupported file type: {ext}")

    def load_bulk_sheet(self):
        cloned = self._clone_file_obj(self.bulk_file)
        excel = pd.ExcelFile(cloned, engine="openpyxl")
        sheet_name = "Sponsored Products Campaigns"
        if sheet_name not in excel.sheet_names:
            sheet_name = excel.sheet_names[0]
        return pd.read_excel(excel, sheet_name=sheet_name, engine="openpyxl", dtype=str)

    def load_sqp_simple_view(self):
        if self.sqp_report_file is None:
            return None
        cloned = self._clone_file_obj(self.sqp_report_file)
        return pd.read_csv(cloned, header=1)

    # -----------------------------
    # GENERAL HELPERS
    # -----------------------------
    def safe_numeric(self, series):
        return pd.to_numeric(series, errors="coerce").fillna(0)

    def clean_text(self, series):
        cleaned = series.fillna("").astype(str).str.strip()
        cleaned = cleaned.replace("nan", "")
        cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
        return cleaned

    def clean_percent_series(self, series):
        cleaned = (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors="coerce").fillna(0)

    def clean_currency_series(self, series):
        cleaned = (
            series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors="coerce").fillna(0)

    def get_first_existing_column(self, df, candidates, label):
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(f"Column for {label} not found. Tried {candidates}. Available: {list(df.columns)}")

    def get_optional_column(self, df, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def combine_preferred_columns(self, df, primary_candidates, fallback_candidates, label):
        primary_col = self.get_optional_column(df, primary_candidates)
        fallback_col = self.get_optional_column(df, fallback_candidates)
        if primary_col is None and fallback_col is None:
            raise KeyError(
                f"Could not find a valid column for {label}. Tried primary={primary_candidates}, "
                f"fallback={fallback_candidates}. Available: {list(df.columns)}"
            )
        primary = self.clean_text(df[primary_col]).replace("", np.nan) if primary_col else pd.Series([np.nan] * len(df), index=df.index)
        fallback = self.clean_text(df[fallback_col]).replace("", np.nan) if fallback_col else pd.Series([np.nan] * len(df), index=df.index)
        return primary.combine_first(fallback).fillna("")

    @staticmethod
    def normalize_text(value: Any) -> str:
        value = str(value or "").strip().lower()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return " ".join(value.split())

    @staticmethod
    def normalize_punct_insensitive(value: Any) -> str:
        value = str(value or "").strip().lower()
        value = re.sub(r"[^a-z0-9]+", "", value)
        return value

    @staticmethod
    def is_asin(value: Any) -> bool:
        return bool(re.fullmatch(r"B0[A-Z0-9]{8}|[A-Z0-9]{10}", str(value or "").strip().upper()))

    def calculate_metrics(self, df):
        df = df.copy()
        spend_col = self.get_optional_column(df, ["Spend"])
        clicks_col = self.get_optional_column(df, ["Clicks"])
        imp_col = self.get_optional_column(df, ["Impressions"])
        orders_col = self.get_optional_column(df, ["7 Day Total Orders (#)", "Orders"])
        sales_col = self.get_optional_column(df, ["7 Day Total Sales ", "Sales", "7 Day Total Sales"])

        if spend_col:
            df[spend_col] = self.safe_numeric(df[spend_col])
        if clicks_col:
            df[clicks_col] = self.safe_numeric(df[clicks_col])
        if imp_col:
            df[imp_col] = self.safe_numeric(df[imp_col])
        if orders_col:
            df[orders_col] = self.safe_numeric(df[orders_col])
        if sales_col:
            df[sales_col] = self.safe_numeric(df[sales_col])

        spend = df[spend_col] if spend_col else 0
        clicks = df[clicks_col] if clicks_col else 0
        impressions = df[imp_col] if imp_col else 0
        orders = df[orders_col] if orders_col else 0
        sales = df[sales_col] if sales_col else 0

        df["roas"] = np.where(spend > 0, sales / spend, 0)
        df["acos"] = np.where(sales > 0, spend / sales, 0)
        df["ctr"] = np.where(impressions > 0, clicks / impressions, 0)
        df["cvr"] = np.where(clicks > 0, orders / clicks, 0)
        df["cpc"] = np.where(clicks > 0, spend / clicks, 0)
        return df

    def _parse_campaign_name(self, campaign_name: str) -> dict:
        parts = [p.strip() for p in str(campaign_name or "").split("|")]
        parts = [p for p in parts if p != ""]
        if not parts:
            return {"description": "", "advertised_item": "", "campaign_type": "", "suffix": ""}
        description = parts[0]
        suffix = parts[-1] if len(parts) >= 2 else ""
        campaign_type = parts[-2] if len(parts) >= 3 else ""
        advertised_item = " | ".join(parts[1:-2]) if len(parts) > 3 else (parts[1] if len(parts) >= 2 else "")
        return {
            "description": description,
            "advertised_item": advertised_item,
            "campaign_type": campaign_type,
            "suffix": suffix,
        }

    def _make_campaign_name(self, description: str, advertised_item: str, campaign_type: str = "SP", suffix: str = "EC") -> str:
        parts = [description.strip(), advertised_item.strip(), campaign_type.strip() or "SP", suffix.strip() or "EC"]
        return " | ".join([p for p in parts if p])

    def _target_key(self, campaign_name: str, ad_group_name: str, target: str, match_type: str) -> str:
        return "||".join(
            [
                self.normalize_text(campaign_name),
                self.normalize_text(ad_group_name),
                self.normalize_punct_insensitive(target),
                self.normalize_text(match_type),
            ]
        )

    def _neg_key(self, campaign_name: str, ad_group_name: str, term: str, match_type: str = "negative exact") -> str:
        return self._target_key(campaign_name, ad_group_name, term, match_type)

    def _parent_key(self, portfolio_name: str, portfolio_id: str, advertised_item: str) -> str:
        if str(portfolio_id or "").strip():
            return f"portfolio_id::{self.normalize_text(portfolio_id)}"
        if str(portfolio_name or "").strip():
            return f"portfolio::{self.normalize_text(portfolio_name)}"
        return ""

    def _new_bid(self, cpc: float) -> float:
        if cpc <= 0:
            return min(self.new_target_bid_cap, 0.5)
        return round(min(cpc * self.new_target_bid_multiplier, self.new_target_bid_cap), 2)

    # -----------------------------
    # LOAD + NORMALIZE REPORTS
    # -----------------------------
    def load_reports(self):
        self.bulk_df = self.load_bulk_sheet()
        self.search_df = self.load_file(self.search_term_file, expected_ext=".xlsx")
        self.targeting_df = self.load_file(self.targeting_file, expected_ext=".xlsx")
        self.impression_share_df = self.load_file(self.impression_share_file, expected_ext=".csv")
        self.business_df = self.load_file(self.business_report_file, expected_ext=".csv") if self.business_report_file is not None else None
        self.sqp_df = self.load_sqp_simple_view() if self.sqp_report_file is not None else None
        self.bulk_columns = list(self.bulk_df.columns)

    def normalize_search_terms(self):
        df = self.calculate_metrics(self.search_df.copy())
        out = pd.DataFrame()
        out["campaign_name"] = self.clean_text(df[self.get_first_existing_column(df, ["Campaign Name"], "campaign")])
        out["ad_group_name"] = self.clean_text(df[self.get_first_existing_column(df, ["Ad Group Name"], "ad group")])
        out["search_term"] = self.clean_text(df[self.get_first_existing_column(df, ["Customer Search Term", "Search Term"], "search term")])
        out["match_type"] = self.clean_text(df[self.get_first_existing_column(df, ["Match Type"], "match type")])
        out["clicks"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Clicks"], "clicks")])
        out["impressions"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Impressions"], "impressions")])
        out["spend"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Spend"], "spend")])
        out["orders"] = self.safe_numeric(df[self.get_first_existing_column(df, ["7 Day Total Orders (#)", "Orders"], "orders")])
        out["sales"] = self.safe_numeric(df[self.get_first_existing_column(df, ["7 Day Total Sales ", "7 Day Total Sales", "Sales"], "sales")])
        out["roas"] = df["roas"]
        out["acos"] = df["acos"]
        out["ctr"] = df["ctr"]
        out["cvr"] = df["cvr"]
        out["cpc"] = df["cpc"]
        out["search_term_key"] = out["search_term"].map(self.normalize_punct_insensitive)
        out["is_asin"] = out["search_term"].map(self.is_asin)
        out = out[out["search_term"] != ""].reset_index(drop=True)
        return out

    def normalize_targeting(self):
        df = self.calculate_metrics(self.targeting_df.copy())
        out = pd.DataFrame()
        out["campaign_name"] = self.clean_text(df[self.get_first_existing_column(df, ["Campaign Name"], "campaign")])
        out["ad_group_name"] = self.clean_text(df[self.get_first_existing_column(df, ["Ad Group Name"], "ad group")])
        out["target"] = self.clean_text(df[self.get_first_existing_column(df, ["Targeting", "Keyword", "Customer Search Term"], "target")])
        out["match_type"] = self.clean_text(df[self.get_first_existing_column(df, ["Match Type"], "match type")])
        out["clicks"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Clicks"], "clicks")])
        out["impressions"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Impressions"], "impressions")])
        out["spend"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Spend"], "spend")])
        out["orders"] = self.safe_numeric(df[self.get_first_existing_column(df, ["7 Day Total Orders (#)", "Orders"], "orders")])
        out["sales"] = self.safe_numeric(df[self.get_first_existing_column(df, ["7 Day Total Sales ", "7 Day Total Sales", "Sales"], "sales")])
        out["roas"] = df["roas"]
        out["acos"] = df["acos"]
        out["ctr"] = df["ctr"]
        out["cvr"] = df["cvr"]
        out["cpc"] = df["cpc"]
        return out

    def normalize_impression_share(self):
        df = self.impression_share_df.copy()
        campaign_col = self.get_first_existing_column(df, ["Campaign Name", "Campaign"], "campaign")
        ad_group_col = self.get_first_existing_column(df, ["Ad Group Name", "Ad Group"], "ad group")
        target_col = self.get_first_existing_column(df, ["Customer Search Term", "Keyword", "Targeting", "Target"], "target")
        match_type_col = self.get_first_existing_column(df, ["Match Type"], "match type")
        share_col = self.get_first_existing_column(
            df,
            ["Search Term Impression Share", "Top-of-search Impression Share", "Search Top Impression Share", "Impression Share"],
            "impression share",
        )
        out = pd.DataFrame()
        out["campaign_name"] = self.clean_text(df[campaign_col])
        out["ad_group_name"] = self.clean_text(df[ad_group_col])
        out["target"] = self.clean_text(df[target_col])
        out["match_type"] = self.clean_text(df[match_type_col])
        out["impression_share_pct"] = self.clean_percent_series(df[share_col])
        return out

    def normalize_bulk_inventory(self):
        df = self.bulk_df.copy()
        entity_col = self.get_first_existing_column(df, ["Entity"], "entity")
        campaign_name = self.combine_preferred_columns(df, ["Campaign Name"], ["Campaign Name (Informational only)"], "campaign name")
        ad_group_name = self.combine_preferred_columns(df, ["Ad Group Name"], ["Ad Group Name (Informational only)"], "ad group name")
        portfolio_name_col = self.get_optional_column(df, ["Portfolio Name", "Portfolio Name (Informational only)"])
        portfolio_id_col = self.get_optional_column(df, ["Portfolio ID", "Portfolio Id"])
        campaign_id_col = self.get_optional_column(df, ["Campaign ID"])
        ad_group_id_col = self.get_optional_column(df, ["Ad Group ID"])
        bid_col = self.get_optional_column(df, ["Bid"])
        keyword_col = self.get_optional_column(df, ["Keyword Text"])
        match_type_col = self.get_optional_column(df, ["Match Type"])
        product_expr_col = self.get_optional_column(
            df,
            [
                "Resolved Product Targeting Expression",
                "Product Targeting Expression",
                "Product Targeting Expression (Informational only)",
                "Resolved Product Targeting Expression (Informational only)",
            ],
        )
        state_col = self.get_optional_column(df, ["State"])
        budget_col = self.get_optional_column(df, ["Daily Budget"])

        inv = pd.DataFrame()
        inv["entity"] = self.clean_text(df[entity_col])
        inv["campaign_name"] = campaign_name
        inv["ad_group_name"] = ad_group_name
        inv["portfolio_name"] = self.clean_text(df[portfolio_name_col]) if portfolio_name_col else ""
        inv["portfolio_id"] = self.clean_text(df[portfolio_id_col]) if portfolio_id_col else ""
        inv["campaign_id"] = self.clean_text(df[campaign_id_col]) if campaign_id_col else ""
        inv["ad_group_id"] = self.clean_text(df[ad_group_id_col]) if ad_group_id_col else ""
        inv["bid"] = self.safe_numeric(df[bid_col]) if bid_col else 0.0
        inv["daily_budget"] = self.safe_numeric(df[budget_col]) if budget_col else 0.0
        inv["match_type"] = self.clean_text(df[match_type_col]) if match_type_col else ""
        inv["keyword_text"] = self.clean_text(df[keyword_col]) if keyword_col else ""
        inv["product_expression"] = self.clean_text(df[product_expr_col]) if product_expr_col else ""
        inv["state"] = self.clean_text(df[state_col]) if state_col else ""

        parsed = inv["campaign_name"].apply(self._parse_campaign_name).apply(pd.Series)
        for col in ["description", "advertised_item", "campaign_type", "suffix"]:
            inv[col] = parsed[col].fillna("")

        inv["parent_key"] = inv.apply(lambda r: self._parent_key(r["portfolio_name"], r["portfolio_id"], r["advertised_item"]), axis=1)
        inv["description_key"] = inv["description"].map(self.normalize_text)
        inv["campaign_key"] = inv["campaign_name"].map(self.normalize_text)
        inv["ad_group_key"] = inv["ad_group_name"].map(self.normalize_text)
        inv["keyword_key"] = inv["keyword_text"].map(self.normalize_punct_insensitive)
        inv["product_key"] = inv["product_expression"].map(self.normalize_punct_insensitive)

        self.bulk_campaign_inventory = inv[inv["entity"] == "Campaign"].drop_duplicates(subset=["campaign_name"]).reset_index(drop=True)
        self.bulk_ad_group_inventory = inv[inv["entity"] == "Ad Group"].drop_duplicates(subset=["campaign_name", "ad_group_name"]).reset_index(drop=True)
        self.bulk_target_inventory = inv[inv["entity"].isin(["Keyword", "Product Targeting", "Negative Keyword"])].reset_index(drop=True)

        self.dest_campaign_inventory = self.bulk_campaign_inventory[
            self.bulk_campaign_inventory["description_key"].str.contains(r"\bdest\b", regex=True)
            | self.bulk_campaign_inventory["description_key"].str.contains(r"\bdestination\b", regex=True)
        ].copy()
        self.dest_campaign_inventory["dest_priority"] = np.where(
            self.dest_campaign_inventory["description_key"].str.contains(r"\bdest\b", regex=True),
            0,
            1,
        )

        self.research_campaign_inventory = self.bulk_campaign_inventory[
            self.bulk_campaign_inventory["description_key"].str.contains(r"\bresearch\b", regex=True)
        ].copy()

        keyword_rows = inv[inv["entity"] == "Keyword"].copy()
        self.existing_exact_keywords = set(
            keyword_rows[keyword_rows["match_type"].str.lower() == "exact"].apply(
                lambda r: self._target_key(r["campaign_name"], r["ad_group_name"], r["keyword_text"], "exact"), axis=1
            )
        )
        self.existing_phrase_keywords = set(
            keyword_rows[keyword_rows["match_type"].str.lower() == "phrase"].apply(
                lambda r: self._target_key(r["campaign_name"], r["ad_group_name"], r["keyword_text"], "phrase"), axis=1
            )
        )
        product_rows = inv[inv["entity"] == "Product Targeting"].copy()
        self.existing_product_targets = set(
            product_rows.apply(lambda r: self._target_key(r["campaign_name"], r["ad_group_name"], r["product_expression"], "product targeting"), axis=1)
        )
        negative_rows = inv[inv["entity"] == "Negative Keyword"].copy()
        self.existing_negative_exact = set(
            negative_rows[negative_rows["match_type"].str.lower() == "negative exact"].apply(
                lambda r: self._neg_key(r["campaign_name"], r["ad_group_name"], r["keyword_text"], "negative exact"), axis=1
            )
        )
        self.existing_ad_groups = set(
            self.bulk_ad_group_inventory.apply(lambda r: f"{self.normalize_text(r['campaign_name'])}||{self.normalize_text(r['ad_group_name'])}", axis=1)
        )
        return inv

    def normalize_bulk_targets(self):
        if self.bulk_target_inventory.empty:
            self.normalize_bulk_inventory()
        out = self.bulk_target_inventory.copy()
        out["target"] = np.where(out["entity"] == "Product Targeting", out["product_expression"], out["keyword_text"])
        out["current_bid"] = out["bid"]
        return out[["entity", "campaign_name", "ad_group_name", "target", "match_type", "current_bid", "campaign_id", "ad_group_id"]].copy()

    def normalize_bulk_campaigns(self):
        if self.bulk_campaign_inventory.empty:
            self.normalize_bulk_inventory()
        return self.bulk_campaign_inventory[["campaign_name", "campaign_id", "daily_budget", "portfolio_name", "portfolio_id", "advertised_item", "campaign_type", "description", "suffix", "parent_key"]].copy()

    # -----------------------------
    # JOINS
    # -----------------------------
    def join_impression_share_to_targeting(self, targeting_df, impression_df):
        left = targeting_df.copy()
        right = impression_df.copy()
        left["join_key"] = left.apply(lambda r: self._target_key(r["campaign_name"], r["ad_group_name"], r["target"], r["match_type"]), axis=1)
        right["join_key"] = right.apply(lambda r: self._target_key(r["campaign_name"], r["ad_group_name"], r["target"], r["match_type"]), axis=1)
        joined = left.merge(right[["join_key", "impression_share_pct"]], on="join_key", how="left")
        joined["impression_share_pct"] = joined["impression_share_pct"].fillna(0)
        return joined

    def join_targeting_to_bulk(self, targeting_df, bulk_df):
        target = targeting_df.copy()
        bulk = bulk_df.copy()
        target["join_key"] = target.apply(lambda r: self._target_key(r["campaign_name"], r["ad_group_name"], r["target"], r["match_type"]), axis=1)
        bulk["join_key"] = bulk.apply(lambda r: self._target_key(r["campaign_name"], r["ad_group_name"], r["target"], r["match_type"]), axis=1)
        joined = target.merge(bulk[["join_key", "current_bid", "campaign_id", "ad_group_id"]], on="join_key", how="left")
        return joined

    # -----------------------------
    # HEALTH / ANALYSIS
    # -----------------------------
    def build_business_sales_total(self):
        if self.business_df is None or not self.enable_tacos_control:
            return None
        df = self.business_df.copy()
        sales_col = self.get_optional_column(
            df,
            ["Ordered Product Sales", "Ordered Product Sales ", "Total Sales", "Sales", "Ordered Product Sales - B2C", "Ordered Product Sales - B2B"],
        )
        if sales_col is None:
            return None
        total_sales = self.clean_currency_series(df[sales_col]).sum()
        return float(total_sales)

    def build_account_health(self, targeting_with_share_df):
        df = targeting_with_share_df.copy()
        total_spend = float(df["spend"].sum())
        total_sales = float(df["sales"].sum())
        account_roas = total_sales / total_spend if total_spend > 0 else 0
        waste_spend = float(df.loc[df["orders"] == 0, "spend"].sum())
        waste_spend_pct = waste_spend / total_spend if total_spend > 0 else 0
        business_total_sales = self.build_business_sales_total()
        tacos = total_spend / business_total_sales if business_total_sales else None
        status = "healthy"
        adjusted_min_roas = self.min_roas
        if account_roas < self.min_roas:
            status = "under_target"
            adjusted_min_roas = self.min_roas * self.account_health_tighten_multiplier
        elif account_roas > self.min_roas * 1.2:
            status = "above_target"
            adjusted_min_roas = self.min_roas * 0.95
        if tacos is not None and tacos > self.max_tacos_target:
            status = "tacos_constrained"
            adjusted_min_roas = max(adjusted_min_roas, self.min_roas * 1.15)
        return {
            "account_roas": round(account_roas, 2),
            "waste_spend": round(waste_spend, 2),
            "waste_spend_pct": round(waste_spend_pct * 100, 2),
            "health_status": status,
            "adjusted_min_roas": round(adjusted_min_roas, 2),
            "tacos_pct": round((tacos or 0) * 100, 2) if tacos is not None else None,
            "tacos_status": "not_used" if tacos is None else ("within_target" if tacos <= self.max_tacos_target else "above_target"),
        }

    def build_campaign_health_dashboard(self, targeting_with_share_df, adjusted_min_roas):
        df = targeting_with_share_df.copy()
        out = (
            df.groupby("campaign_name", as_index=False)
            .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"), spend=("spend", "sum"), sales=("sales", "sum"), orders=("orders", "sum"), avg_impression_share_pct=("impression_share_pct", "mean"))
        )
        out["roas"] = np.where(out["spend"] > 0, out["sales"] / out["spend"], 0)
        out["acos"] = np.where(out["sales"] > 0, out["spend"] / out["sales"], 0)
        conditions = [
            (out["spend"] >= 100) & (out["orders"] == 0),
            (out["roas"] < adjusted_min_roas) & (out["clicks"] >= self.min_clicks),
            (out["roas"] >= adjusted_min_roas * 1.15) & (out["orders"] >= 3) & (out["avg_impression_share_pct"] < 20),
        ]
        out["campaign_status"] = np.select(conditions, ["Waste Alert", "Under Target", "Scalable"], default="Stable")
        out["avg_impression_share_pct"] = out["avg_impression_share_pct"].round(2)
        out["roas"] = out["roas"].round(2)
        out["acos"] = (out["acos"] * 100).round(2)
        return out.sort_values(by=["spend", "sales"], ascending=[False, False]).reset_index(drop=True)

    def build_pre_run_preview(self, bid_recommendations, search_term_actions, campaign_budget_actions):
        preview = {
            "bid_updates": int(len(bid_recommendations)),
            "bid_increases": int((bid_recommendations.get("optimizer_action", pd.Series(dtype=str)) == "INCREASE_BID").sum()) if not bid_recommendations.empty else 0,
            "bid_decreases": int((bid_recommendations.get("optimizer_action", pd.Series(dtype=str)) == "DECREASE_BID").sum()) if not bid_recommendations.empty else 0,
            "search_term_actions": int(len(search_term_actions)),
            "budget_actions": int(len(campaign_budget_actions)),
            "budget_increases": int((campaign_budget_actions.get("optimizer_action", pd.Series(dtype=str)) == "INCREASE_BUDGET").sum()) if not campaign_budget_actions.empty else 0,
            "budget_decreases": int((campaign_budget_actions.get("optimizer_action", pd.Series(dtype=str)) == "DECREASE_BUDGET").sum()) if not campaign_budget_actions.empty else 0,
        }
        if not search_term_actions.empty:
            preview["graduations"] = int(search_term_actions["optimizer_action"].isin(["ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST", "NEGATE_SOURCE_ONLY"]).sum())
            preview["negative_exacts"] = int(((search_term_actions["source_negative_required"]) | (search_term_actions["optimizer_action"] == "ADD_NEGATIVE_EXACT")).sum())
            preview["dest_exact_adds"] = int((search_term_actions["optimizer_action"] == "ADD_TO_DEST_EXACT").sum())
            preview["research_phrase_adds"] = int((search_term_actions["optimizer_action"] == "ADD_TO_RESEARCH_PHRASE").sum())
            preview["asin_target_adds"] = int((search_term_actions["optimizer_action"] == "ADD_ASIN_TO_DEST").sum())
        else:
            preview["graduations"] = 0
            preview["negative_exacts"] = 0
            preview["dest_exact_adds"] = 0
            preview["research_phrase_adds"] = 0
            preview["asin_target_adds"] = 0
        return preview

    def build_smart_warnings(self, targeting_with_share_df, search_terms_df, campaign_health_df, account_health, adjusted_min_roas, sqp_summary=None):
        warnings = []
        suggestions = []
        if account_health.get("health_status") in {"under_target", "tacos_constrained"}:
            warnings.append("Account efficiency is below target. Review bid raises carefully and prioritize waste cleanup.")
        if not search_terms_df.empty and (search_terms_df["orders"] >= self.graduation_orders_threshold).any():
            suggestions.append("Graduation opportunities exist. Review destination and research routing actions.")
        if not campaign_health_df.empty and (campaign_health_df["campaign_status"] == "Waste Alert").any():
            warnings.append("Some campaigns have significant spend without orders.")
        if sqp_summary and sqp_summary.get("uploaded"):
            suggestions.append("Search Query Performance data is available for additional keyword expansion ideas.")
        return {"warnings": warnings, "suggestions": suggestions}

    # -----------------------------
    # SEARCH QUERY PERFORMANCE
    # -----------------------------
    def normalize_sqp(self):
        if self.sqp_df is None or self.sqp_df.empty:
            return pd.DataFrame()
        df = self.sqp_df.copy()
        out = pd.DataFrame()
        out["search_query"] = self.clean_text(df[self.get_first_existing_column(df, ["Search Query"], "search query")])
        out["search_query_score"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Search Query Score"], "search query score")])
        out["search_query_volume"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Search Query Volume"], "volume")])
        out["purchase_rate_pct"] = self.clean_percent_series(df[self.get_first_existing_column(df, ["Purchases: Purchase Rate %"], "purchase rate")])
        out["purchases_total_count"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Purchases: Total Count"], "purchase count")])
        share_col = self.get_optional_column(df, ["Purchases: Brand Share %"])
        out["purchases_brand_share_pct"] = self.clean_percent_series(df[share_col]) if share_col else 0
        return out[out["search_query"] != ""].reset_index(drop=True)

    def build_sqp_opportunities(self, sqp_df, search_terms_df):
        if sqp_df is None or sqp_df.empty:
            return pd.DataFrame(), {"uploaded": False}
        sqp = sqp_df.copy()
        search_set = set(search_terms_df["search_term"].map(self.normalize_text).tolist()) if search_terms_df is not None and not search_terms_df.empty else set()
        sqp["in_search_term_report"] = sqp["search_query"].map(self.normalize_text).isin(search_set)
        sqp["opportunity_tier"] = np.select(
            [
                (sqp["search_query_volume"] >= 1000) & (sqp["purchase_rate_pct"] >= 6) & (sqp["purchases_total_count"] >= 25) & (sqp["purchases_brand_share_pct"] < 20),
                (sqp["search_query_volume"] >= 500) & (sqp["purchase_rate_pct"] >= 4) & (sqp["purchases_total_count"] >= 10),
            ],
            ["High Opportunity", "Monitor"],
            default="Low Priority",
        )
        sqp["recommended_action"] = np.select(
            [
                (sqp["opportunity_tier"] == "High Opportunity") & (~sqp["in_search_term_report"]),
                (sqp["opportunity_tier"] == "High Opportunity") & (sqp["in_search_term_report"]),
                (sqp["opportunity_tier"] == "Monitor") & (~sqp["in_search_term_report"]),
            ],
            ["Test as Exact / Phrase", "Prioritize current query for routing", "Monitor and test selectively"],
            default="No immediate action",
        )
        summary = {
            "uploaded": True,
            "total_queries": int(len(sqp)),
            "high_opportunity": int((sqp["opportunity_tier"] == "High Opportunity").sum()),
            "monitor": int((sqp["opportunity_tier"] == "Monitor").sum()),
            "low_priority": int((sqp["opportunity_tier"] == "Low Priority").sum()),
            "harvest_overlap": int(((sqp["opportunity_tier"] == "High Opportunity") & (sqp["in_search_term_report"])).sum()),
        }
        return sqp.sort_values(by=["search_query_volume", "purchases_total_count"], ascending=[False, False]).reset_index(drop=True), summary

    # -----------------------------
    # BID / BUDGET ACTIONS
    # -----------------------------
    def build_bid_recommendations(self, targeting_with_share_df, joined_targeting_df, adjusted_min_roas):
        if not self.enable_bid_updates:
            return pd.DataFrame()
        df = joined_targeting_df.copy()
        if df.empty:
            return df
        df = df[df["current_bid"] > 0].copy()
        if df.empty:
            return pd.DataFrame()
        actions = []
        for _, r in df.iterrows():
            action = None
            recommended_bid = None
            rationale = ""
            if r["orders"] >= self.min_orders_for_scaling and r["roas"] >= adjusted_min_roas * self.scale_roas_multiplier and r.get("impression_share_pct", 0) < 20:
                action = "INCREASE_BID"
                recommended_bid = min(r["current_bid"] * (1 + self.max_bid_up), self.max_bid_cap)
                rationale = "High-efficiency target with room to scale."
            elif r["clicks"] >= self.min_clicks and ((r["orders"] == 0 and self.should_zero_order_decrease_bid()) or (r["orders"] > 0 and r["roas"] < adjusted_min_roas)):
                action = "DECREASE_BID"
                recommended_bid = max(r["current_bid"] * (1 - self.max_bid_down), 0.02)
                rationale = "Underperforming target relative to account efficiency goals."
            if action:
                actions.append(
                    {
                        "campaign_name": r["campaign_name"],
                        "ad_group_name": r["ad_group_name"],
                        "target": r["target"],
                        "match_type": r["match_type"],
                        "clicks": r["clicks"],
                        "orders": r["orders"],
                        "spend": r["spend"],
                        "sales": r["sales"],
                        "roas": round(r["roas"], 2),
                        "impression_share_pct": round(float(r.get("impression_share_pct", 0)), 2),
                        "current_bid": round(float(r["current_bid"]), 2),
                        "recommended_bid": round(float(recommended_bid), 2),
                        "optimizer_action": action,
                        "rationale": rationale,
                    }
                )
        return pd.DataFrame(actions)

    def should_zero_order_negate(self):
        return self.zero_order_action in ["Add Negative", "Both"]

    def should_zero_order_decrease_bid(self):
        return self.zero_order_action in ["Decrease Bid", "Both"]

    def build_campaign_budget_actions(self, targeting_with_share_df, bulk_campaigns_df, adjusted_min_roas):
        if not self.enable_budget_updates:
            return pd.DataFrame()
        perf = (
            targeting_with_share_df.groupby("campaign_name", as_index=False)
            .agg(clicks=("clicks", "sum"), spend=("spend", "sum"), sales=("sales", "sum"), orders=("orders", "sum"), avg_impression_share_pct=("impression_share_pct", "mean"))
        )
        perf["roas"] = np.where(perf["spend"] > 0, perf["sales"] / perf["spend"], 0)
        joined = perf.merge(bulk_campaigns_df[["campaign_name", "daily_budget"]], on="campaign_name", how="left")
        actions = []
        for _, r in joined.iterrows():
            budget = float(r.get("daily_budget", 0) or 0)
            if budget <= 0:
                continue
            action = None
            new_budget = None
            rationale = ""
            if r["orders"] >= 3 and r["roas"] >= adjusted_min_roas * 1.15 and r["avg_impression_share_pct"] < 20:
                action = "INCREASE_BUDGET"
                new_budget = min(budget * (1 + self.budget_up_pct), self.max_budget_cap)
                rationale = "Profitable campaign with impression-share headroom."
            elif r["clicks"] >= self.min_clicks and r["roas"] < adjusted_min_roas:
                action = "DECREASE_BUDGET"
                new_budget = max(budget * (1 - self.budget_down_pct), 1.0)
                rationale = "Campaign under target efficiency."
            if action:
                actions.append(
                    {
                        "campaign_name": r["campaign_name"],
                        "clicks": r["clicks"],
                        "orders": r["orders"],
                        "spend": round(float(r["spend"]), 2),
                        "sales": round(float(r["sales"]), 2),
                        "roas": round(float(r["roas"]), 2),
                        "avg_impression_share_pct": round(float(r["avg_impression_share_pct"]), 2),
                        "daily_budget": round(budget, 2),
                        "recommended_daily_budget": round(float(new_budget), 2),
                        "optimizer_action": action,
                        "rationale": rationale,
                    }
                )
        return pd.DataFrame(actions)

    # -----------------------------
    # DESTINATION / RESEARCH LOOKUPS
    # -----------------------------
    def _get_source_campaign_meta(self, campaign_name: str) -> Optional[pd.Series]:
        if self.bulk_campaign_inventory.empty:
            return None
        matches = self.bulk_campaign_inventory[self.bulk_campaign_inventory["campaign_name"] == campaign_name]
        if not matches.empty:
            return matches.iloc[0]
        parsed = self._parse_campaign_name(campaign_name)
        parent_key = ""
        return pd.Series({
            "campaign_name": campaign_name,
            "portfolio_name": "",
            "portfolio_id": "",
            "advertised_item": parsed["advertised_item"],
            "campaign_type": parsed["campaign_type"] or "SP",
            "description": parsed["description"],
            "suffix": parsed["suffix"] or "EC",
            "parent_key": parent_key,
        })

    def _find_existing_campaign(self, parent_key: str, advertised_item: str, campaign_kind: str) -> Optional[pd.Series]:
        if campaign_kind == "dest":
            df = self.dest_campaign_inventory.copy()
            if df.empty:
                return None
            df = df[df["advertised_item"].map(self.normalize_text) == self.normalize_text(advertised_item)]
            if parent_key:
                df = df[df["parent_key"] == parent_key]
            if df.empty:
                return None
            df = df.sort_values(by=["dest_priority", "campaign_name"], ascending=[True, True])
            return df.iloc[0]
        if campaign_kind == "research":
            df = self.research_campaign_inventory.copy()
            if df.empty:
                return None
            df = df[df["advertised_item"].map(self.normalize_text) == self.normalize_text(advertised_item)]
            if parent_key:
                df = df[df["parent_key"] == parent_key]
            if df.empty:
                return None
            return df.sort_values(by=["campaign_name"]).iloc[0]
        return None

    def _build_campaign_stub(self, source_meta: pd.Series, campaign_kind: str) -> dict:
        description = "Dest" if campaign_kind == "dest" else "Research"
        campaign_name = self._make_campaign_name(description, source_meta.get("advertised_item", ""), source_meta.get("campaign_type", "SP"), source_meta.get("suffix", "EC"))
        return {
            "campaign_name": campaign_name,
            "portfolio_name": source_meta.get("portfolio_name", ""),
            "portfolio_id": source_meta.get("portfolio_id", ""),
            "advertised_item": source_meta.get("advertised_item", ""),
            "campaign_type": source_meta.get("campaign_type", "SP") or "SP",
            "description": description,
            "suffix": source_meta.get("suffix", "EC") or "EC",
            "parent_key": source_meta.get("parent_key", ""),
        }

    def _dest_default_ad_group(self, source_ad_group: str, search_term: str, is_asin: bool) -> str:
        if is_asin:
            return "ASIN Targets"
        if source_ad_group and self.normalize_text(source_ad_group) not in {"asin targets", "default ad group"}:
            return source_ad_group
        cleaned = str(search_term or "").strip()
        return cleaned[:100] if cleaned else "Default Ad Group"

    def _keyword_exists(self, campaign_name: str, ad_group_name: str, term: str, match_type: str) -> bool:
        key = self._target_key(campaign_name, ad_group_name, term, match_type)
        if self.normalize_text(match_type) == "exact":
            return key in self.existing_exact_keywords
        if self.normalize_text(match_type) == "phrase":
            return key in self.existing_phrase_keywords
        return False

    def _product_target_exists(self, campaign_name: str, ad_group_name: str, expr: str) -> bool:
        key = self._target_key(campaign_name, ad_group_name, expr, "product targeting")
        return key in self.existing_product_targets

    def _negative_exists(self, campaign_name: str, ad_group_name: str, term: str) -> bool:
        return self._neg_key(campaign_name, ad_group_name, term, "negative exact") in self.existing_negative_exact

    def _ad_group_exists(self, campaign_name: str, ad_group_name: str) -> bool:
        return f"{self.normalize_text(campaign_name)}||{self.normalize_text(ad_group_name)}" in self.existing_ad_groups

    def _can_graduate(self, source_meta: pd.Series) -> bool:
        return bool(str(source_meta.get("parent_key", "")).strip())

    # -----------------------------
    # SEARCH TERM ROUTING ENGINE
    # -----------------------------
    def build_search_term_actions(self, search_terms_df, adjusted_min_roas=None):
        if not self.enable_search_harvesting and not self.enable_negative_keywords:
            return pd.DataFrame()

        rows = []
        for _, r in search_terms_df.iterrows():
            source_meta = self._get_source_campaign_meta(r["campaign_name"])
            can_graduate = self._can_graduate(source_meta)
            is_asin = bool(r["is_asin"])
            qualifies_for_dest = r["orders"] >= self.graduation_orders_threshold and r["acos"] <= self.dest_acos_threshold
            qualifies_for_research = (
                r["orders"] >= self.graduation_orders_threshold
                and not qualifies_for_dest
                and (
                    (self.research_ctr_min <= r["ctr"] < self.research_ctr_max)
                    or (self.research_cvr_min <= r["cvr"] < self.research_cvr_max)
                )
            )
            qualifies_as_loser = (
                r["clicks"] > self.loser_clicks_threshold
                and r["ctr"] < self.loser_ctr_threshold
                and r["cvr"] < self.loser_cvr_threshold
            )

            action = "NO_ACTION"
            target_entity_type = ""
            target_match_type = ""
            dest_campaign_name = ""
            dest_ad_group_name = ""
            create_campaign = False
            create_ad_group = False
            source_negative_required = False
            already_exists = False
            reason = ""

            if qualifies_as_loser and self.enable_negative_keywords:
                action = "ADD_NEGATIVE_EXACT"
                target_entity_type = "negative_keyword"
                target_match_type = "negative exact"
                reason = "Low CTR and low CVR after sufficient clicks."
            elif can_graduate and self.enable_search_harvesting and (qualifies_for_dest or qualifies_for_research):
                if qualifies_for_dest:
                    campaign_kind = "dest"
                    existing_campaign = self._find_existing_campaign(source_meta.get("parent_key", ""), source_meta.get("advertised_item", ""), campaign_kind)
                    campaign_stub = self._build_campaign_stub(source_meta, campaign_kind) if existing_campaign is None else existing_campaign.to_dict()
                    dest_campaign_name = campaign_stub["campaign_name"]
                    dest_ad_group_name = self._dest_default_ad_group(r["ad_group_name"], r["search_term"], is_asin)
                    create_campaign = existing_campaign is None
                    create_ad_group = not self._ad_group_exists(dest_campaign_name, dest_ad_group_name)
                    if is_asin:
                        action = "ADD_ASIN_TO_DEST"
                        target_entity_type = "product_target"
                        target_match_type = "product targeting"
                        already_exists = self._product_target_exists(dest_campaign_name, dest_ad_group_name, r["search_term"])
                        reason = "Winning ASIN target meets destination efficiency threshold."
                    else:
                        action = "ADD_TO_DEST_EXACT"
                        target_entity_type = "keyword"
                        target_match_type = "exact"
                        already_exists = self._keyword_exists(dest_campaign_name, dest_ad_group_name, r["search_term"], "exact")
                        reason = "Winning search term meets destination exact criteria."
                else:
                    campaign_kind = "research"
                    existing_campaign = self._find_existing_campaign(source_meta.get("parent_key", ""), source_meta.get("advertised_item", ""), campaign_kind)
                    campaign_stub = self._build_campaign_stub(source_meta, campaign_kind) if existing_campaign is None else existing_campaign.to_dict()
                    dest_campaign_name = campaign_stub["campaign_name"]
                    dest_ad_group_name = self._dest_default_ad_group(r["ad_group_name"], r["search_term"], False)
                    create_campaign = existing_campaign is None
                    create_ad_group = not self._ad_group_exists(dest_campaign_name, dest_ad_group_name)
                    action = "ADD_TO_RESEARCH_PHRASE"
                    target_entity_type = "keyword"
                    target_match_type = "phrase"
                    already_exists = self._keyword_exists(dest_campaign_name, dest_ad_group_name, r["search_term"], "phrase")
                    reason = "Converting term needs more proof before destination exact."

                source_negative_required = True
                if already_exists:
                    action = "NEGATE_SOURCE_ONLY"
                    create_campaign = False
                    create_ad_group = False
                    reason = "Target already exists in destination structure, so only source negation is needed."
            elif (r["orders"] >= self.graduation_orders_threshold) and not can_graduate:
                reason = "Skipped graduation because the source campaign has no reliable portfolio / parent-group routing key."

            if action == "ADD_NEGATIVE_EXACT" and self._negative_exists(r["campaign_name"], r["ad_group_name"], r["search_term"]):
                action = "NO_ACTION"
                reason = "Negative exact already exists in the source ad group."

            rows.append(
                {
                    "campaign_name": r["campaign_name"],
                    "ad_group_name": r["ad_group_name"],
                    "search_term": r["search_term"],
                    "source_match_type": r["match_type"],
                    "clicks": int(r["clicks"]),
                    "impressions": int(r["impressions"]),
                    "spend": round(float(r["spend"]), 2),
                    "orders": int(r["orders"]),
                    "sales": round(float(r["sales"]), 2),
                    "roas": round(float(r["roas"]), 2),
                    "acos": round(float(r["acos"] * 100), 2),
                    "ctr": round(float(r["ctr"] * 100), 4),
                    "cvr": round(float(r["cvr"] * 100), 4),
                    "cpc": round(float(r["cpc"]), 2),
                    "is_asin": is_asin,
                    "optimizer_action": action,
                    "target_entity_type": target_entity_type,
                    "target_match_type": target_match_type,
                    "destination_campaign_name": dest_campaign_name,
                    "destination_ad_group_name": dest_ad_group_name,
                    "create_destination_campaign": bool(create_campaign),
                    "create_destination_ad_group": bool(create_ad_group),
                    "destination_already_contains_target": bool(already_exists),
                    "source_negative_required": bool(source_negative_required or action == "ADD_NEGATIVE_EXACT"),
                    "starting_bid": self._new_bid(float(r["cpc"])),
                    "reason": reason,
                }
            )

        actions = pd.DataFrame(rows)
        if actions.empty:
            return actions
        actions = actions[actions["optimizer_action"] != "NO_ACTION"].reset_index(drop=True)
        return actions

    # -----------------------------
    # BULK UPDATE GENERATION
    # -----------------------------
    def _blank_bulk_row(self) -> dict:
        row = {col: "" for col in self.bulk_columns}
        for col in [
            "Entity",
            "Operation",
            "Campaign Name",
            "Ad Group Name",
            "Portfolio Name",
            "Portfolio ID",
            "Campaign ID",
            "Ad Group ID",
            "Keyword Text",
            "Product Targeting Expression",
            "Resolved Product Targeting Expression",
            "Match Type",
            "Bid",
            "Daily Budget",
            "State",
            "Optimizer Action",
            "Reason",
            "Source Campaign Name",
            "Source Ad Group Name",
            "Start Date",
        ]:
            row.setdefault(col, "")
        return row

    def _make_campaign_row(self, campaign_stub: dict, daily_budget: float = 25.0) -> dict:
        row = self._blank_bulk_row()
        row["Entity"] = "Campaign"
        row["Operation"] = "Create"
        row["Campaign Name"] = campaign_stub["campaign_name"]
        row["Portfolio Name"] = campaign_stub.get("portfolio_name", "")
        row["Portfolio ID"] = campaign_stub.get("portfolio_id", "")
        row["Daily Budget"] = round(float(daily_budget), 2)
        row["State"] = "enabled"
        row["Optimizer Action"] = "CREATE_CAMPAIGN"
        row["Reason"] = "Auto-created by graduation engine."
        row["Start Date"] = datetime.utcnow().strftime("%Y%m%d")
        return row

    def _make_ad_group_row(self, campaign_name: str, ad_group_name: str, default_bid: float = 0.75) -> dict:
        row = self._blank_bulk_row()
        row["Entity"] = "Ad Group"
        row["Operation"] = "Create"
        row["Campaign Name"] = campaign_name
        row["Ad Group Name"] = ad_group_name
        row["Bid"] = round(float(default_bid), 2)
        row["State"] = "enabled"
        row["Optimizer Action"] = "CREATE_AD_GROUP"
        row["Reason"] = "Auto-created by graduation engine."
        return row

    def _make_keyword_row(self, campaign_name: str, ad_group_name: str, keyword_text: str, match_type: str, bid: float, optimizer_action: str, reason: str) -> dict:
        row = self._blank_bulk_row()
        row["Entity"] = "Keyword"
        row["Operation"] = "Create"
        row["Campaign Name"] = campaign_name
        row["Ad Group Name"] = ad_group_name
        row["Keyword Text"] = keyword_text
        row["Match Type"] = match_type
        row["Bid"] = round(float(bid), 2)
        row["State"] = "enabled"
        row["Optimizer Action"] = optimizer_action
        row["Reason"] = reason
        return row

    def _make_product_target_row(self, campaign_name: str, ad_group_name: str, expression: str, bid: float, optimizer_action: str, reason: str) -> dict:
        row = self._blank_bulk_row()
        row["Entity"] = "Product Targeting"
        row["Operation"] = "Create"
        row["Campaign Name"] = campaign_name
        row["Ad Group Name"] = ad_group_name
        row["Product Targeting Expression"] = expression
        row["Resolved Product Targeting Expression"] = expression
        row["Bid"] = round(float(bid), 2)
        row["State"] = "enabled"
        row["Optimizer Action"] = optimizer_action
        row["Reason"] = reason
        return row

    def _make_negative_row(self, campaign_name: str, ad_group_name: str, keyword_text: str, reason: str) -> dict:
        row = self._blank_bulk_row()
        row["Entity"] = "Negative Keyword"
        row["Operation"] = "Create"
        row["Campaign Name"] = campaign_name
        row["Ad Group Name"] = ad_group_name
        row["Keyword Text"] = keyword_text
        row["Match Type"] = "Negative Exact"
        row["State"] = "enabled"
        row["Optimizer Action"] = "ADD_NEGATIVE_EXACT"
        row["Reason"] = reason
        return row

    def generate_bid_bulk_updates(self, bid_recommendations):
        if bid_recommendations.empty:
            return pd.DataFrame()
        rows = []
        for _, r in bid_recommendations.iterrows():
            row = self._blank_bulk_row()
            row["Entity"] = "Keyword"
            row["Operation"] = "Update"
            row["Campaign Name"] = r["campaign_name"]
            row["Ad Group Name"] = r["ad_group_name"]
            row["Keyword Text"] = r["target"]
            row["Match Type"] = r["match_type"]
            row["Bid"] = r["recommended_bid"]
            row["Optimizer Action"] = r["optimizer_action"]
            row["Reason"] = r["rationale"]
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_budget_bulk_updates(self, campaign_budget_actions):
        if campaign_budget_actions.empty:
            return pd.DataFrame()
        rows = []
        for _, r in campaign_budget_actions.iterrows():
            row = self._blank_bulk_row()
            row["Entity"] = "Campaign"
            row["Operation"] = "Update"
            row["Campaign Name"] = r["campaign_name"]
            row["Daily Budget"] = r["recommended_daily_budget"]
            row["Optimizer Action"] = r["optimizer_action"]
            row["Reason"] = r["rationale"]
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_search_term_bulk_updates(self, search_term_actions):
        if search_term_actions.empty:
            return pd.DataFrame()
        rows = []
        created_campaigns = set()
        created_ad_groups = set()

        for _, r in search_term_actions.iterrows():
            action = r["optimizer_action"]
            campaign_name = r["destination_campaign_name"]
            ad_group_name = r["destination_ad_group_name"]

            if action in {"ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST"}:
                source_meta = self._get_source_campaign_meta(r["campaign_name"])
                campaign_kind = "dest" if action in {"ADD_TO_DEST_EXACT", "ADD_ASIN_TO_DEST"} else "research"
                campaign_stub = self._build_campaign_stub(source_meta, campaign_kind)
                if r["create_destination_campaign"]:
                    ck = self.normalize_text(campaign_name)
                    if ck not in created_campaigns:
                        rows.append(self._make_campaign_row(campaign_stub))
                        created_campaigns.add(ck)
                if r["create_destination_ad_group"]:
                    agk = f"{self.normalize_text(campaign_name)}||{self.normalize_text(ad_group_name)}"
                    if agk not in created_ad_groups:
                        rows.append(self._make_ad_group_row(campaign_name, ad_group_name, r["starting_bid"]))
                        created_ad_groups.add(agk)

                if action in {"ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE"}:
                    rows.append(
                        self._make_keyword_row(
                            campaign_name=campaign_name,
                            ad_group_name=ad_group_name,
                            keyword_text=r["search_term"],
                            match_type=r["target_match_type"].title(),
                            bid=r["starting_bid"],
                            optimizer_action=action,
                            reason=r["reason"],
                        )
                    )
                elif action == "ADD_ASIN_TO_DEST":
                    rows.append(
                        self._make_product_target_row(
                            campaign_name=campaign_name,
                            ad_group_name=ad_group_name,
                            expression=r["search_term"],
                            bid=r["starting_bid"],
                            optimizer_action=action,
                            reason=r["reason"],
                        )
                    )

            if bool(r["source_negative_required"]):
                if not self._negative_exists(r["campaign_name"], r["ad_group_name"], r["search_term"]):
                    neg_reason = "Negate source after graduation." if action != "ADD_NEGATIVE_EXACT" else r["reason"]
                    rows.append(self._make_negative_row(r["campaign_name"], r["ad_group_name"], r["search_term"], neg_reason))
                    self.existing_negative_exact.add(self._neg_key(r["campaign_name"], r["ad_group_name"], r["search_term"], "negative exact"))

        return pd.DataFrame(rows)

    def apply_final_safeguards(self, combined_bulk_updates):
        if combined_bulk_updates.empty:
            return combined_bulk_updates
        df = combined_bulk_updates.copy()
        df["_dedupe_key"] = (
            df["Entity"].astype(str)
            + "||"
            + df["Campaign Name"].astype(str)
            + "||"
            + df.get("Ad Group Name", "").astype(str)
            + "||"
            + df.get("Keyword Text", "").astype(str)
            + "||"
            + df.get("Product Targeting Expression", "").astype(str)
            + "||"
            + df.get("Match Type", "").astype(str)
            + "||"
            + df["Operation"].astype(str)
        )
        df = df.drop_duplicates(subset=["_dedupe_key"]).drop(columns=["_dedupe_key"]).reset_index(drop=True)
        preferred_order = [
            "Campaign",
            "Ad Group",
            "Keyword",
            "Product Targeting",
            "Negative Keyword",
        ]
        df["_entity_sort"] = pd.Categorical(df["Entity"], categories=preferred_order, ordered=True)
        df = df.sort_values(by=["Campaign Name", "Ad Group Name", "_entity_sort"]).drop(columns=["_entity_sort"]).reset_index(drop=True)
        return df

    def build_simulation_summary(self, combined_bulk_updates, account_health):
        df = combined_bulk_updates.copy() if not combined_bulk_updates.empty else pd.DataFrame()
        action_counts = df["Optimizer Action"].value_counts().to_dict() if not df.empty else {}
        return {
            "total_actions": int(len(df)),
            "campaign_creates": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "CREATE_CAMPAIGN").sum()) if not df.empty else 0,
            "ad_group_creates": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "CREATE_AD_GROUP").sum()) if not df.empty else 0,
            "dest_exact_adds": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "ADD_TO_DEST_EXACT").sum()) if not df.empty else 0,
            "research_phrase_adds": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "ADD_TO_RESEARCH_PHRASE").sum()) if not df.empty else 0,
            "asin_target_adds": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "ADD_ASIN_TO_DEST").sum()) if not df.empty else 0,
            "negative_exact_adds": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "ADD_NEGATIVE_EXACT").sum()) if not df.empty else 0,
            "bid_updates": int((df.get("Optimizer Action", pd.Series(dtype=str)).isin(["INCREASE_BID", "DECREASE_BID"])).sum()) if not df.empty else 0,
            "bid_increases": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "INCREASE_BID").sum()) if not df.empty else 0,
            "bid_decreases": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "DECREASE_BID").sum()) if not df.empty else 0,
            "budget_updates": int((df.get("Optimizer Action", pd.Series(dtype=str)).isin(["INCREASE_BUDGET", "DECREASE_BUDGET"])).sum()) if not df.empty else 0,
            "budget_increases": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "INCREASE_BUDGET").sum()) if not df.empty else 0,
            "budget_decreases": int((df.get("Optimizer Action", pd.Series(dtype=str)) == "DECREASE_BUDGET").sum()) if not df.empty else 0,
            "account_health": account_health,
            "action_breakdown": action_counts,
        }

    # -----------------------------
    # RUN HISTORY
    # -----------------------------
    def load_run_history(self):
        if not self.run_history_path.exists():
            return []
        try:
            return json.loads(self.run_history_path.read_text())
        except Exception:
            return []

    def save_run_history(self, simulation_summary, account_health):
        history = self.load_run_history()
        history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "total_actions": simulation_summary.get("total_actions", 0),
                "dest_exact_adds": simulation_summary.get("dest_exact_adds", 0),
                "research_phrase_adds": simulation_summary.get("research_phrase_adds", 0),
                "asin_target_adds": simulation_summary.get("asin_target_adds", 0),
                "negative_exact_adds": simulation_summary.get("negative_exact_adds", 0),
                "account_roas": account_health.get("account_roas", 0),
            }
        )
        history = history[-25:]
        try:
            self.run_history_path.write_text(json.dumps(history, indent=2))
        except Exception:
            pass

    # -----------------------------
    # PUBLIC ENTRYPOINTS
    # -----------------------------
    def analyze(self):
        self.load_reports()
        search_terms = self.normalize_search_terms()
        targeting = self.normalize_targeting()
        impression_share = self.normalize_impression_share()
        self.normalize_bulk_inventory()
        bulk_targets = self.normalize_bulk_targets()
        bulk_campaigns = self.normalize_bulk_campaigns()
        sqp = self.normalize_sqp()

        targeting_with_share = self.join_impression_share_to_targeting(targeting, impression_share)
        joined_targeting = self.join_targeting_to_bulk(targeting, bulk_targets)
        account_health = self.build_account_health(targeting_with_share)
        adjusted_min_roas = account_health["adjusted_min_roas"]
        bid_recommendations = self.build_bid_recommendations(targeting_with_share, joined_targeting, adjusted_min_roas)
        search_term_actions = self.build_search_term_actions(search_terms, adjusted_min_roas)
        campaign_budget_actions = self.build_campaign_budget_actions(targeting_with_share, bulk_campaigns, adjusted_min_roas)
        campaign_health_dashboard = self.build_campaign_health_dashboard(targeting_with_share, adjusted_min_roas)
        sqp_opportunities, sqp_summary = self.build_sqp_opportunities(sqp, search_terms)
        smart = self.build_smart_warnings(targeting_with_share, search_terms, campaign_health_dashboard, account_health, adjusted_min_roas, sqp_summary)
        pre_run_preview = self.build_pre_run_preview(bid_recommendations, search_term_actions, campaign_budget_actions)
        account_summary = {
            "total_spend": round(float(targeting_with_share["spend"].sum()), 2),
            "total_sales": round(float(targeting_with_share["sales"].sum()), 2),
            "campaigns_under_target": int((campaign_health_dashboard["campaign_status"] == "Under Target").sum()),
            "campaigns_scalable": int((campaign_health_dashboard["campaign_status"] == "Scalable").sum()),
            "campaigns_waste_alert": int((campaign_health_dashboard["campaign_status"] == "Waste Alert").sum()),
        }
        return {
            "account_health": account_health,
            "account_summary": account_summary,
            "campaign_health_dashboard": campaign_health_dashboard,
            "smart_warnings": smart["warnings"],
            "optimization_suggestions": smart["suggestions"],
            "pre_run_preview": pre_run_preview,
            "sqp_opportunities": sqp_opportunities,
            "sqp_summary": sqp_summary,
            "search_term_actions": search_term_actions,
            "bid_recommendations": bid_recommendations,
            "campaign_budget_actions": campaign_budget_actions,
        }

    def process(self):
        self.load_reports()
        search_terms = self.normalize_search_terms()
        targeting = self.normalize_targeting()
        impression_share = self.normalize_impression_share()
        self.normalize_bulk_inventory()
        bulk_targets = self.normalize_bulk_targets()
        bulk_campaigns = self.normalize_bulk_campaigns()
        sqp = self.normalize_sqp()

        targeting_with_share = self.join_impression_share_to_targeting(targeting, impression_share)
        joined_targeting = self.join_targeting_to_bulk(targeting, bulk_targets)
        account_health = self.build_account_health(targeting_with_share)
        adjusted_min_roas = account_health["adjusted_min_roas"]

        bid_recommendations = self.build_bid_recommendations(targeting_with_share, joined_targeting, adjusted_min_roas)
        search_term_actions = self.build_search_term_actions(search_terms, adjusted_min_roas)
        campaign_budget_actions = self.build_campaign_budget_actions(targeting_with_share, bulk_campaigns, adjusted_min_roas)

        bid_bulk_updates = self.generate_bid_bulk_updates(bid_recommendations)
        search_bulk_updates = self.generate_search_term_bulk_updates(search_term_actions)
        budget_bulk_updates = self.generate_budget_bulk_updates(campaign_budget_actions)

        combined_bulk_updates = pd.concat([bid_bulk_updates, search_bulk_updates, budget_bulk_updates], ignore_index=True)
        combined_bulk_updates = self.apply_final_safeguards(combined_bulk_updates)
        simulation_summary = self.build_simulation_summary(combined_bulk_updates, account_health)

        campaign_health_dashboard = self.build_campaign_health_dashboard(targeting_with_share, adjusted_min_roas)
        sqp_opportunities, sqp_summary = self.build_sqp_opportunities(sqp, search_terms)
        smart = self.build_smart_warnings(targeting_with_share, search_terms, campaign_health_dashboard, account_health, adjusted_min_roas, sqp_summary)
        pre_run_preview = self.build_pre_run_preview(bid_recommendations, search_term_actions, campaign_budget_actions)
        account_summary = {
            "total_spend": round(float(targeting_with_share["spend"].sum()), 2),
            "total_sales": round(float(targeting_with_share["sales"].sum()), 2),
            "campaigns_under_target": int((campaign_health_dashboard["campaign_status"] == "Under Target").sum()),
            "campaigns_scalable": int((campaign_health_dashboard["campaign_status"] == "Scalable").sum()),
            "campaigns_waste_alert": int((campaign_health_dashboard["campaign_status"] == "Waste Alert").sum()),
        }
        self.save_run_history(simulation_summary, account_health)
        return {
            "bid_recommendations": bid_recommendations,
            "search_term_actions": search_term_actions,
            "campaign_budget_actions": campaign_budget_actions,
            "combined_bulk_updates": combined_bulk_updates,
            "account_health": account_health,
            "simulation_summary": simulation_summary,
            "campaign_health_dashboard": campaign_health_dashboard,
            "smart_warnings": smart["warnings"],
            "optimization_suggestions": smart["suggestions"],
            "pre_run_preview": pre_run_preview,
            "account_summary": account_summary,
            "run_history": self.load_run_history(),
            "sqp_opportunities": sqp_opportunities,
            "sqp_summary": sqp_summary,
        }
