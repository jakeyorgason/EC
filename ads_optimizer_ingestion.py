import io
import os
import calendar
from datetime import date

import numpy as np
import pandas as pd


class AdsOptimizerEngine:
    def __init__(
        self,
        bulk_file,
        search_term_file,
        targeting_file,
        impression_share_file,
        business_report_file=None,
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
    ):
        self.bulk_file = bulk_file
        self.search_term_file = search_term_file
        self.targeting_file = targeting_file
        self.impression_share_file = impression_share_file
        self.business_report_file = business_report_file

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

        self.existing_any_keywords = set()
        self.existing_negative_keywords = set()
        self.keyword_capable_ad_groups = set()

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
            try:
                current_pos = file_obj.tell()
            except Exception:
                current_pos = None

            try:
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                data = file_obj.read()
            finally:
                try:
                    if current_pos is not None and hasattr(file_obj, "seek"):
                        file_obj.seek(current_pos)
                except Exception:
                    pass

            return io.BytesIO(data)

        raise ValueError("Unsupported file object type.")

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
        return pd.read_excel(
            cloned,
            sheet_name="Sponsored Products Campaigns",
            engine="openpyxl",
            dtype=str,
        )

    # -----------------------------
    # HELPERS
    # -----------------------------
    def safe_numeric(self, series):
        return pd.to_numeric(series, errors="coerce").fillna(0)

    def clean_text(self, series):
        cleaned = series.fillna("").astype(str).str.strip()
        cleaned = cleaned.replace("nan", "")
        cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
        return cleaned

    def get_first_existing_column(self, df, candidates, label):
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(
            f"Column for {label} not found. Tried {candidates}. Available: {list(df.columns)}"
        )

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
                f"Could not find a valid column for {label}. "
                f"Tried primary={primary_candidates}, fallback={fallback_candidates}. "
                f"Available: {list(df.columns)}"
            )

        if primary_col is not None:
            primary_series = self.clean_text(df[primary_col]).replace("", np.nan)
        else:
            primary_series = pd.Series([np.nan] * len(df), index=df.index)

        if fallback_col is not None:
            fallback_series = self.clean_text(df[fallback_col]).replace("", np.nan)
        else:
            fallback_series = pd.Series([np.nan] * len(df), index=df.index)

        return primary_series.combine_first(fallback_series).fillna("")

    def blank_series(self, df):
        return pd.Series([""] * len(df), index=df.index)

    def should_zero_order_negate(self):
        return self.zero_order_action in ["Add Negative", "Both"]

    def should_zero_order_decrease_bid(self):
        return self.zero_order_action in ["Decrease Bid", "Both"]

    # -----------------------------
    # CAMPAIGN HEALTH DASHBOARD
    # -----------------------------
    def build_campaign_health_dashboard(self, targeting_with_share_df, adjusted_min_roas):
        df = targeting_with_share_df.copy()

        campaign_health = (
            df.groupby("campaign_name", as_index=False)
            .agg(
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum"),
                spend=("spend", "sum"),
                sales=("sales", "sum"),
                orders=("orders", "sum"),
                avg_impression_share_pct=("impression_share_pct", "mean"),
            )
        )

        campaign_health["roas"] = np.where(
            campaign_health["spend"] > 0,
            campaign_health["sales"] / campaign_health["spend"],
            0,
        )

        campaign_health["acos"] = np.where(
            campaign_health["sales"] > 0,
            campaign_health["spend"] / campaign_health["sales"],
            0,
        )

        conditions = [
            (campaign_health["spend"] >= 100) & (campaign_health["orders"] == 0),
            (campaign_health["roas"] < adjusted_min_roas) & (campaign_health["clicks"] >= self.min_clicks),
            (campaign_health["roas"] >= adjusted_min_roas * 1.15)
            & (campaign_health["orders"] >= 3)
            & (campaign_health["avg_impression_share_pct"] < 20),
        ]

        choices = ["Waste Alert", "Under Target", "Scalable"]

        campaign_health["campaign_status"] = np.select(
            conditions,
            choices,
            default="Stable",
        )

        campaign_health["avg_impression_share_pct"] = campaign_health["avg_impression_share_pct"].round(2)
        campaign_health["roas"] = campaign_health["roas"].round(2)
        campaign_health["acos"] = (campaign_health["acos"] * 100).round(2)

        campaign_health = campaign_health.sort_values(
            by=["spend", "sales"],
            ascending=[False, False],
        ).reset_index(drop=True)

        return campaign_health

    # -----------------------------
    # SMART WARNINGS + SUGGESTIONS
    # -----------------------------
    def build_smart_warnings(
        self,
        targeting_with_share_df,
        search_terms_df,
        campaign_health_df,
        account_health,
        adjusted_min_roas,
    ):
        warnings = []
        suggestions = []

        waste_campaigns = campaign_health_df[
            (campaign_health_df["spend"] >= 100) & (campaign_health_df["orders"] == 0)
        ]

        scalable_campaigns = campaign_health_df[
            (campaign_health_df["roas"] >= adjusted_min_roas * 1.15)
            & (campaign_health_df["orders"] >= 3)
            & (campaign_health_df["avg_impression_share_pct"] < 20)
        ]

        high_click_zero_order_terms = search_terms_df[
            (search_terms_df["clicks"] >= self.zero_order_click_threshold)
            & (search_terms_df["orders"] == 0)
        ]

        harvest_candidates = search_terms_df[
            (search_terms_df["orders"] >= 4)
            & (search_terms_df["clicks"] >= 5)
            & (search_terms_df["roas"] >= max(adjusted_min_roas, self.min_roas))
            & (search_terms_df["match_type"].str.lower() != "exact")
        ]

        pacing = self.build_budget_pacing_status()

        if len(waste_campaigns) > 0:
            warnings.append(
                f"{len(waste_campaigns)} campaign(s) have at least $100 spend and 0 orders."
            )
            suggestions.append(
                "Prioritize bid decreases and waste cleanup in zero-order spend campaigns."
            )

        if len(scalable_campaigns) > 0:
            warnings.append(
                f"{len(scalable_campaigns)} campaign(s) have strong ROAS with low impression share."
            )
            suggestions.append(
                "These campaigns may have safe scaling headroom through bids or budgets."
            )

        if len(high_click_zero_order_terms) > 0:
            warnings.append(
                f"{len(high_click_zero_order_terms)} search term(s) have reached the losing-click threshold with 0 orders."
            )
            suggestions.append(
                "Review zero-order search terms for negatives and bid suppression."
            )

        if len(harvest_candidates) > 0:
            warnings.append(
                f"{len(harvest_candidates)} search term(s) qualify as likely harvest candidates."
            )
            suggestions.append(
                "Harvest proven converting search terms into Exact where appropriate."
            )

        if account_health.get("health_status") == "under_target":
            warnings.append(
                f"Account ROAS is below target at {account_health.get('account_roas')}."
            )
            suggestions.append(
                "Keep scaling selective and prioritize efficiency improvements first."
            )

        if account_health.get("tacos_status") == "above_target":
            warnings.append(
                "Account TACOS is above the configured guardrail."
            )
            suggestions.append(
                "Tighten scaling and monitor spend relative to total sales."
            )

        if pacing.get("enabled") and pacing.get("over_pace"):
            warnings.append(
                "Monthly budget pacing is currently over target."
            )
            suggestions.append(
                "Suppress scaling until monthly pace returns to target."
            )

        if not warnings:
            warnings.append("No major risk flags detected from the current uploaded data.")

        if not suggestions:
            suggestions.append("The account appears stable. Continue using balanced optimization settings.")

        return {
            "warnings": warnings[:6],
            "suggestions": suggestions[:6],
        }

    # -----------------------------
    # PRE-RUN ACTION PREVIEW
    # -----------------------------
    def build_pre_run_preview(
        self,
        bid_recommendations,
        search_term_actions,
        campaign_budget_actions,
    ):
        return {
            "bid_increases": int((bid_recommendations["recommended_action"] == "INCREASE_BID").sum()),
            "bid_decreases": int((bid_recommendations["recommended_action"] == "DECREASE_BID").sum()),
            "negatives_added": int((search_term_actions["search_term_action"] == "ADD_NEGATIVE_PHRASE").sum()),
            "harvested_keywords": int((search_term_actions["search_term_action"] == "HARVEST_TO_EXACT").sum()),
            "budget_increases": int((campaign_budget_actions["campaign_action"] == "INCREASE_BUDGET").sum()),
            "budget_decreases": int((campaign_budget_actions["campaign_action"] == "DECREASE_BUDGET").sum()),
        }

    # -----------------------------
    # RUN HISTORY
    # -----------------------------
    def save_run_history(self, simulation_summary, account_health):
        history_path = "run_history.csv"

        row = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy_mode": self.strategy_mode,
            "min_roas": self.min_roas,
            "min_clicks": self.min_clicks,
            "zero_order_click_threshold": self.zero_order_click_threshold,
            "zero_order_action": self.zero_order_action,
            "enable_tacos_control": self.enable_tacos_control,
            "enable_monthly_budget_control": self.enable_monthly_budget_control,
            "account_roas": account_health.get("account_roas"),
            "tacos_pct": account_health.get("tacos_pct"),
            "health_status": account_health.get("health_status"),
            "bid_increases": simulation_summary.get("bid_increases"),
            "bid_decreases": simulation_summary.get("bid_decreases"),
            "negatives_added": simulation_summary.get("negatives_added"),
            "harvested_keywords": simulation_summary.get("harvested_keywords"),
            "budget_increases": simulation_summary.get("budget_increases"),
            "budget_decreases": simulation_summary.get("budget_decreases"),
            "estimated_spend_impact_pct": simulation_summary.get("estimated_spend_impact_pct"),
        }

        new_row_df = pd.DataFrame([row])

        if os.path.exists(history_path):
            existing = pd.read_csv(history_path)
            updated = pd.concat([existing, new_row_df], ignore_index=True)
        else:
            updated = new_row_df

        updated.to_csv(history_path, index=False)

    def load_run_history(self):
        history_path = "run_history.csv"
        if os.path.exists(history_path):
            return pd.read_csv(history_path)
        return pd.DataFrame()

    # -----------------------------
    # PRE-OPTIMIZATION DIAGNOSTICS
    # -----------------------------
    def analyze(self):
        self.load_reports()

        search_terms = self.normalize_search_terms()
        targeting = self.normalize_targeting()
        impression_share = self.normalize_impression_share()
        bulk_targets = self.normalize_bulk_targets()
        bulk_campaigns = self.normalize_bulk_campaigns()

        targeting_with_share = self.join_impression_share_to_targeting(
            targeting,
            impression_share,
        )

        joined_targeting = self.join_targeting_to_bulk(
            targeting,
            bulk_targets,
        )

        account_health = self.build_account_health(targeting_with_share)
        adjusted_min_roas = account_health["adjusted_min_roas"]

        bid_recommendations = self.build_bid_recommendations(
            targeting_with_share,
            joined_targeting,
            adjusted_min_roas,
        )

        search_term_actions = self.build_search_term_actions(
            search_terms,
            adjusted_min_roas,
        )

        campaign_budget_actions = self.build_campaign_budget_actions(
            targeting_with_share,
            bulk_campaigns,
            adjusted_min_roas,
        )

        campaign_health_dashboard = self.build_campaign_health_dashboard(
            targeting_with_share,
            adjusted_min_roas,
        )

        smart = self.build_smart_warnings(
            targeting_with_share_df=targeting_with_share,
            search_terms_df=search_terms,
            campaign_health_df=campaign_health_dashboard,
            account_health=account_health,
            adjusted_min_roas=adjusted_min_roas,
        )

        pre_run_preview = self.build_pre_run_preview(
            bid_recommendations=bid_recommendations,
            search_term_actions=search_term_actions,
            campaign_budget_actions=campaign_budget_actions,
        )

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
        }

    # -----------------------------
    # LOAD REPORTS
    # -----------------------------
    def load_reports(self):
        self.bulk_df = self.load_bulk_sheet()

        self.search_df = self.load_file(self.search_term_file, expected_ext=".xlsx")
        self.targeting_df = self.load_file(self.targeting_file, expected_ext=".xlsx")
        self.impression_share_df = self.load_file(self.impression_share_file, expected_ext=".csv")
        self.business_df = (
            self.load_file(self.business_report_file, expected_ext=".csv")
            if self.business_report_file is not None
            else None
        )

    # -----------------------------
    # METRIC CALCULATIONS
    # -----------------------------
    def calculate_metrics(self, df):
        df = df.copy()

        numeric_cols = [
            "Spend",
            "Clicks",
            "Impressions",
            "7 Day Total Orders (#)",
            "7 Day Total Sales ",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = self.safe_numeric(df[col])

        if "Spend" in df.columns and "7 Day Total Sales " in df.columns:
            df["roas"] = np.where(df["Spend"] > 0, df["7 Day Total Sales "] / df["Spend"], 0)
            df["acos"] = np.where(df["7 Day Total Sales "] > 0, df["Spend"] / df["7 Day Total Sales "], 0)

        if "Impressions" in df.columns and "Clicks" in df.columns:
            df["ctr"] = np.where(df["Impressions"] > 0, df["Clicks"] / df["Impressions"], 0)

        if "Clicks" in df.columns and "7 Day Total Orders (#)" in df.columns:
            df["cvr"] = np.where(df["Clicks"] > 0, df["7 Day Total Orders (#)"] / df["Clicks"], 0)

        if "Clicks" in df.columns and "Spend" in df.columns:
            df["cpc"] = np.where(df["Clicks"] > 0, df["Spend"] / df["Clicks"], 0)

        return df

    # -----------------------------
    # NORMALIZE SEARCH TERMS
    # -----------------------------
    def normalize_search_terms(self):
        df = self.calculate_metrics(self.search_df.copy())

        normalized = pd.DataFrame()
        normalized["campaign_name"] = self.clean_text(df["Campaign Name"])
        normalized["ad_group_name"] = self.clean_text(df["Ad Group Name"])
        normalized["search_term"] = self.clean_text(df["Customer Search Term"])
        normalized["match_type"] = self.clean_text(df["Match Type"])

        normalized["clicks"] = self.safe_numeric(df["Clicks"])
        normalized["impressions"] = self.safe_numeric(df["Impressions"])
        normalized["spend"] = self.safe_numeric(df["Spend"])
        normalized["orders"] = self.safe_numeric(df["7 Day Total Orders (#)"])
        normalized["sales"] = self.safe_numeric(df["7 Day Total Sales "])

        normalized["roas"] = df["roas"]
        normalized["acos"] = df["acos"]
        normalized["ctr"] = df["ctr"]
        normalized["cvr"] = df["cvr"]
        normalized["cpc"] = df["cpc"]

        return normalized

    # -----------------------------
    # NORMALIZE TARGETING
    # -----------------------------
    def normalize_targeting(self):
        df = self.calculate_metrics(self.targeting_df.copy())

        normalized = pd.DataFrame()
        normalized["campaign_name"] = self.clean_text(df["Campaign Name"])
        normalized["ad_group_name"] = self.clean_text(df["Ad Group Name"])
        normalized["target"] = self.clean_text(df["Targeting"])
        normalized["match_type"] = self.clean_text(df["Match Type"])

        normalized["clicks"] = self.safe_numeric(df["Clicks"])
        normalized["impressions"] = self.safe_numeric(df["Impressions"])
        normalized["spend"] = self.safe_numeric(df["Spend"])
        normalized["orders"] = self.safe_numeric(df["7 Day Total Orders (#)"])
        normalized["sales"] = self.safe_numeric(df["7 Day Total Sales "])

        normalized["roas"] = df["roas"]
        normalized["acos"] = df["acos"]
        normalized["ctr"] = df["ctr"]
        normalized["cvr"] = df["cvr"]
        normalized["cpc"] = df["cpc"]

        return normalized

    # -----------------------------
    # NORMALIZE IMPRESSION SHARE
    # -----------------------------
    def normalize_impression_share(self):
        df = self.impression_share_df.copy()

        campaign_col = self.get_first_existing_column(df, ["Campaign Name", "Campaign"], "campaign")
        ad_group_col = self.get_first_existing_column(df, ["Ad Group Name", "Ad Group"], "ad group")
        target_col = self.get_first_existing_column(
            df,
            ["Customer Search Term", "Keyword", "Targeting", "Target"],
            "target",
        )
        match_type_col = self.get_first_existing_column(df, ["Match Type"], "match type")
        share_col = self.get_first_existing_column(
            df,
            [
                "Search Term Impression Share",
                "Top-of-search Impression Share",
                "Search Top Impression Share",
                "Impression Share",
            ],
            "impression share",
        )

        normalized = pd.DataFrame()
        normalized["campaign_name"] = self.clean_text(df[campaign_col])
        normalized["ad_group_name"] = self.clean_text(df[ad_group_col])
        normalized["target"] = self.clean_text(df[target_col])
        normalized["match_type"] = self.clean_text(df[match_type_col])

        raw_share = (
            df[share_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
            .str.strip()
        )

        normalized["impression_share_pct"] = pd.to_numeric(raw_share, errors="coerce").fillna(0)
        return normalized

    # -----------------------------
    # NORMALIZE BULK TARGETS
    # -----------------------------
    def normalize_bulk_targets(self):
        df = self.bulk_df.copy()

        entity_col = self.get_first_existing_column(df, ["Entity"], "entity")
        match_type_col = self.get_first_existing_column(df, ["Match Type"], "match type")
        bid_col = self.get_first_existing_column(df, ["Bid"], "bid")
        keyword_col = self.get_first_existing_column(df, ["Keyword Text"], "keyword text")

        negative_keyword_text_col = self.get_optional_column(df, ["Keyword Text"])
        negative_match_type_col = self.get_optional_column(df, ["Match Type"])

        product_expr_col = self.get_optional_column(
            df,
            [
                "Resolved Product Targeting Expression",
                "Product Targeting Expression",
                "Product Targeting Expression (Informational only)",
                "Resolved Product Targeting Expression (Informational only)",
            ],
        )

        keyword_id_col = self.get_optional_column(df, ["Keyword ID"])
        campaign_id_col = self.get_optional_column(df, ["Campaign ID"])
        ad_group_id_col = self.get_optional_column(df, ["Ad Group ID"])

        rows = df[df[entity_col].isin(["Keyword", "Product Targeting"])].copy()

        normalized = pd.DataFrame()
        normalized["entity"] = self.clean_text(rows[entity_col])
        normalized["campaign_name"] = self.combine_preferred_columns(
            rows, ["Campaign Name"], ["Campaign Name (Informational only)"], "campaign name"
        )
        normalized["ad_group_name"] = self.combine_preferred_columns(
            rows, ["Ad Group Name"], ["Ad Group Name (Informational only)"], "ad group name"
        )
        normalized["match_type"] = self.clean_text(rows[match_type_col])
        normalized["current_bid"] = self.safe_numeric(rows[bid_col])

        normalized["keyword_id"] = self.clean_text(rows[keyword_id_col]) if keyword_id_col else ""
        normalized["campaign_id"] = self.clean_text(rows[campaign_id_col]) if campaign_id_col else ""
        normalized["ad_group_id"] = self.clean_text(rows[ad_group_id_col]) if ad_group_id_col else ""

        keyword_text = self.clean_text(rows[keyword_col])

        if product_expr_col is not None:
            product_expr = self.clean_text(rows[product_expr_col])
        else:
            product_expr = self.blank_series(rows)

        normalized["target"] = np.where(
            normalized["entity"] == "Keyword",
            keyword_text,
            product_expr,
        )

        existing_keywords = df[df[entity_col] == "Keyword"].copy()
        self.existing_any_keywords = set(
            (
                self.combine_preferred_columns(
                    existing_keywords,
                    ["Campaign Name"],
                    ["Campaign Name (Informational only)"],
                    "campaign name",
                ).str.lower().str.strip()
                + "||"
                + self.combine_preferred_columns(
                    existing_keywords,
                    ["Ad Group Name"],
                    ["Ad Group Name (Informational only)"],
                    "ad group name",
                ).str.lower().str.strip()
                + "||"
                + self.clean_text(existing_keywords[keyword_col])
                .str.lower()
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            ).tolist()
        )

        existing_negatives = df[df[entity_col] == "Negative Keyword"].copy()
        if len(existing_negatives) > 0 and negative_keyword_text_col and negative_match_type_col:
            self.existing_negative_keywords = set(
                (
                    self.combine_preferred_columns(
                        existing_negatives,
                        ["Campaign Name"],
                        ["Campaign Name (Informational only)"],
                        "campaign name",
                    ).str.lower().str.strip()
                    + "||"
                    + self.combine_preferred_columns(
                        existing_negatives,
                        ["Ad Group Name"],
                        ["Ad Group Name (Informational only)"],
                        "ad group name",
                    ).str.lower().str.strip()
                    + "||"
                    + self.clean_text(existing_negatives[negative_keyword_text_col])
                    .str.lower()
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    + "||"
                    + self.clean_text(existing_negatives[negative_match_type_col])
                    .str.lower()
                    .str.strip()
                ).tolist()
            )
        else:
            self.existing_negative_keywords = set()

        keyword_ad_groups = df[df[entity_col] == "Keyword"].copy()
        self.keyword_capable_ad_groups = set(
            (
                self.combine_preferred_columns(
                    keyword_ad_groups,
                    ["Campaign Name"],
                    ["Campaign Name (Informational only)"],
                    "campaign name",
                ).str.lower().str.strip()
                + "||"
                + self.combine_preferred_columns(
                    keyword_ad_groups,
                    ["Ad Group Name"],
                    ["Ad Group Name (Informational only)"],
                    "ad group name",
                ).str.lower().str.strip()
            ).tolist()
        )

        return normalized

    # -----------------------------
    # NORMALIZE BULK CAMPAIGNS
    # -----------------------------
    def normalize_bulk_campaigns(self):
        df = self.bulk_df.copy()

        entity_col = self.get_first_existing_column(df, ["Entity"], "entity")
        campaign_id_col = self.get_optional_column(df, ["Campaign ID"])
        budget_col = self.get_optional_column(df, ["Daily Budget"])

        rows = df[df[entity_col] == "Campaign"].copy()

        normalized = pd.DataFrame()
        normalized["campaign_name"] = self.combine_preferred_columns(
            rows, ["Campaign Name"], ["Campaign Name (Informational only)"], "campaign name"
        )
        normalized["campaign_id"] = self.clean_text(rows[campaign_id_col]) if campaign_id_col else ""
        normalized["daily_budget"] = self.safe_numeric(rows[budget_col]) if budget_col else 0

        return normalized

    # -----------------------------
    # BUSINESS REPORT / TACOS
    # -----------------------------
    def build_business_sales_total(self):
        if self.business_df is None or not self.enable_tacos_control:
            return None

        df = self.business_df.copy()

        possible_sales_cols = [
            "Ordered Product Sales",
            "Ordered Product Sales ",
            "Total Sales",
            "Sales",
            "Ordered Product Sales - B2C",
            "Ordered Product Sales - B2B",
        ]

        sales_col = self.get_optional_column(df, possible_sales_cols)
        if sales_col is None:
            return None

        sales_series = (
            df[sales_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        total_sales = pd.to_numeric(sales_series, errors="coerce").fillna(0).sum()
        return float(total_sales)

    # -----------------------------
    # ACCOUNT HEALTH
    # -----------------------------
    def build_account_health(self, targeting_with_share_df):
        df = targeting_with_share_df.copy()

        total_spend = df["spend"].sum()
        total_sales = df["sales"].sum()

        account_roas = total_sales / total_spend if total_spend > 0 else 0
        waste_spend = df.loc[df["orders"] == 0, "spend"].sum()
        waste_spend_pct = waste_spend / total_spend if total_spend > 0 else 0

        business_total_sales = self.build_business_sales_total()
        tacos = None
        tacos_status = "not_used"

        if business_total_sales is not None and business_total_sales > 0:
            tacos = total_spend / business_total_sales
            tacos_status = "within_target" if tacos <= self.max_tacos_target else "above_target"

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
            "tacos_pct": round(tacos * 100, 2) if tacos is not None else None,
            "tacos_status": tacos_status,
        }

    # -----------------------------
    # MONTHLY BUDGET PACING
    # -----------------------------
    def build_budget_pacing_status(self):
        if not self.enable_monthly_budget_control or self.monthly_account_budget <= 0:
            return {
                "enabled": False,
                "over_pace": False,
                "remaining_budget": None,
                "allowed_daily_pace": None,
                "current_daily_pace": None,
            }

        today = date.today()
        days_in_month = calendar.monthrange(today.year, today.month)[1]
        remaining_days = max(days_in_month - today.day + 1, 1)

        remaining_budget = self.monthly_account_budget - self.month_to_date_spend
        allowed_daily_pace = remaining_budget / remaining_days if remaining_days > 0 else 0
        current_daily_pace = self.month_to_date_spend / max(today.day, 1)

        pacing_limit = allowed_daily_pace * (1 + self.pacing_buffer_pct)
        over_pace = current_daily_pace > pacing_limit

        return {
            "enabled": True,
            "over_pace": over_pace,
            "remaining_budget": remaining_budget,
            "allowed_daily_pace": allowed_daily_pace,
            "current_daily_pace": current_daily_pace,
        }

    # -----------------------------
    # JOINS
    # -----------------------------
    def join_impression_share_to_targeting(self, targeting_df, impression_df):
        targeting = targeting_df.copy()
        impression = impression_df.copy()

        targeting["join_key"] = (
            targeting["campaign_name"].str.lower().str.strip()
            + "||"
            + targeting["ad_group_name"].str.lower().str.strip()
            + "||"
            + targeting["target"].str.lower().str.strip()
            + "||"
            + targeting["match_type"].str.lower().str.strip()
        )

        impression["join_key"] = (
            impression["campaign_name"].str.lower().str.strip()
            + "||"
            + impression["ad_group_name"].str.lower().str.strip()
            + "||"
            + impression["target"].str.lower().str.strip()
            + "||"
            + impression["match_type"].str.lower().str.strip()
        )

        joined = targeting.merge(
            impression[["join_key", "impression_share_pct"]],
            on="join_key",
            how="left",
        )

        joined["impression_share_pct"] = joined["impression_share_pct"].fillna(0)
        return joined

    def join_targeting_to_bulk(self, targeting_df, bulk_df):
        targeting = targeting_df.copy()
        bulk = bulk_df.copy()

        targeting["join_key"] = (
            targeting["campaign_name"].str.lower().str.strip()
            + "||"
            + targeting["ad_group_name"].str.lower().str.strip()
            + "||"
            + targeting["target"].str.lower().str.strip()
            + "||"
            + targeting["match_type"].str.lower().str.strip()
        )

        bulk["join_key"] = (
            bulk["campaign_name"].str.lower().str.strip()
            + "||"
            + bulk["ad_group_name"].str.lower().str.strip()
            + "||"
            + bulk["target"].str.lower().str.strip()
            + "||"
            + bulk["match_type"].str.lower().str.strip()
        )

        return targeting.merge(
            bulk[["join_key", "entity", "current_bid", "keyword_id", "campaign_id", "ad_group_id"]],
            on="join_key",
            how="left",
        )

    # -----------------------------
    # BID RECOMMENDATIONS
    # -----------------------------
    def build_bid_recommendations(self, targeting_with_share_df, joined_targeting_df, adjusted_min_roas):
        perf = targeting_with_share_df.copy()
        bids = joined_targeting_df[
            [
                "campaign_name",
                "ad_group_name",
                "target",
                "match_type",
                "current_bid",
                "keyword_id",
                "campaign_id",
                "ad_group_id",
            ]
        ].copy()

        recs = perf.merge(
            bids,
            on=["campaign_name", "ad_group_name", "target", "match_type"],
            how="left",
        )

        recs["current_bid"] = recs["current_bid"].fillna(0)

        pacing = self.build_budget_pacing_status()
        over_pace = pacing["over_pace"]

        actions = []
        recommended_bids = []

        for _, row in recs.iterrows():
            action = "NO_ACTION"
            new_bid = row["current_bid"]

            if row["current_bid"] <= 0 or not self.enable_bid_updates:
                actions.append(action)
                recommended_bids.append(new_bid)
                continue

            if (
                self.should_zero_order_decrease_bid()
                and row["clicks"] >= self.zero_order_click_threshold
                and row["orders"] == 0
            ):
                action = "DECREASE_BID"
                new_bid = round(max(row["current_bid"] * (1 - self.max_bid_down), 0.02), 2)

            elif row["roas"] < adjusted_min_roas and row["clicks"] >= self.min_clicks:
                action = "DECREASE_BID"
                new_bid = round(max(row["current_bid"] * (1 - self.max_bid_down), 0.02), 2)

            elif (
                not over_pace
                and row["roas"] > adjusted_min_roas * self.scale_roas_multiplier
                and row["orders"] >= self.min_orders_for_scaling
            ):
                action = "INCREASE_BID"
                new_bid = round(row["current_bid"] * (1 + self.max_bid_up), 2)

            new_bid = min(new_bid, self.max_bid_cap)
            actions.append(action)
            recommended_bids.append(new_bid)

        recs["recommended_action"] = actions
        recs["recommended_bid"] = recommended_bids
        return recs

    # -----------------------------
    # SEARCH TERM ACTIONS
    # -----------------------------
    def build_search_term_actions(self, search_terms_df, adjusted_min_roas):
        df = search_terms_df.copy()

        targeting_lookup = (
            self.normalize_bulk_targets()[["campaign_name", "ad_group_name", "campaign_id", "ad_group_id"]]
            .drop_duplicates(subset=["campaign_name", "ad_group_name"])
        )

        df = df.merge(
            targeting_lookup,
            on=["campaign_name", "ad_group_name"],
            how="left",
        )

        actions = []
        recommended_bids = []

        for _, row in df.iterrows():
            action = "NO_ACTION"
            rec_bid = min(max(round(row["cpc"] * 1.10, 2), 0.20), self.max_bid_cap)

            campaign_name = str(row["campaign_name"]).strip()
            ad_group_name = str(row["ad_group_name"]).strip()
            search_term = str(row["search_term"]).strip().lower()
            match_type = str(row["match_type"]).strip().lower()

            campaign_name_l = campaign_name.lower()
            ad_group_name_l = ad_group_name.lower()
            normalized_term = " ".join(search_term.split())

            is_auto_ad_group = (
                "auto" in campaign_name_l
                or "auto" in ad_group_name_l
                or match_type in ["close-match", "loose-match", "substitutes", "complements"]
            )

            ad_group_key = campaign_name_l.strip() + "||" + ad_group_name_l.strip()

            keyword_exists_key = (
                campaign_name_l.strip()
                + "||"
                + ad_group_name_l.strip()
                + "||"
                + normalized_term
            )

            negative_key = (
                campaign_name_l.strip()
                + "||"
                + ad_group_name_l.strip()
                + "||"
                + normalized_term
                + "||negative phrase"
            )

            if (
                self.enable_negative_keywords
                and self.should_zero_order_negate()
                and row["clicks"] >= max(self.zero_order_click_threshold, 20)
                and row["orders"] == 0
                and row["sales"] == 0
                and normalized_term != ""
                and negative_key not in self.existing_negative_keywords
                and len(normalized_term.split()) >= 2
                and not any(term in normalized_term for term in ["anchor straps"])
            ):
                action = "ADD_NEGATIVE_PHRASE"

            elif (
                self.enable_search_harvesting
                and not is_auto_ad_group
                and ad_group_key in self.keyword_capable_ad_groups
                and row["orders"] >= 4
                and row["clicks"] >= 5
                and row["roas"] >= max(adjusted_min_roas, self.min_roas)
                and match_type != "exact"
                and normalized_term != ""
                and keyword_exists_key not in self.existing_any_keywords
            ):
                action = "HARVEST_TO_EXACT"

            actions.append(action)
            recommended_bids.append(rec_bid)

        df["search_term_action"] = actions
        df["recommended_bid"] = recommended_bids
        return df

    # -----------------------------
    # CAMPAIGN BUDGET ACTIONS
    # -----------------------------
    def build_campaign_budget_actions(self, targeting_with_share_df, bulk_campaigns_df, adjusted_min_roas):
        campaign_perf = (
            targeting_with_share_df.groupby("campaign_name", as_index=False).agg(
                clicks=("clicks", "sum"),
                orders=("orders", "sum"),
                spend=("spend", "sum"),
                sales=("sales", "sum"),
                avg_impression_share_pct=("impression_share_pct", "mean"),
            )
        )

        campaign_perf["roas"] = np.where(
            campaign_perf["spend"] > 0,
            campaign_perf["sales"] / campaign_perf["spend"],
            0,
        )

        recs = campaign_perf.merge(
            bulk_campaigns_df,
            on="campaign_name",
            how="left",
        )

        pacing = self.build_budget_pacing_status()
        over_pace = pacing["over_pace"]

        actions = []
        recommended_budgets = []

        for _, row in recs.iterrows():
            action = "NO_ACTION"
            new_budget = row["daily_budget"]

            if row["daily_budget"] <= 0 or not self.enable_budget_updates:
                actions.append(action)
                recommended_budgets.append(new_budget)
                continue

            if (
                not over_pace
                and row["roas"] >= adjusted_min_roas * 1.15
                and row["orders"] >= 3
                and row["avg_impression_share_pct"] < 20
            ):
                action = "INCREASE_BUDGET"
                new_budget = round(row["daily_budget"] * (1 + self.budget_up_pct), 2)

            elif row["roas"] < adjusted_min_roas and row["clicks"] >= max(self.min_clicks * 3, 25):
                action = "DECREASE_BUDGET"
                new_budget = round(max(row["daily_budget"] * (1 - self.budget_down_pct), 1.00), 2)

            new_budget = min(new_budget, self.max_budget_cap)
            actions.append(action)
            recommended_budgets.append(new_budget)

        recs["campaign_action"] = actions
        recs["recommended_daily_budget"] = recommended_budgets
        return recs

    # -----------------------------
    # BULK GENERATORS
    # -----------------------------
    def generate_bid_bulk_updates(self, recommendations_df):
        actionable = recommendations_df[
            (recommendations_df["recommended_action"] != "NO_ACTION")
            & (recommendations_df["current_bid"] > 0)
        ].copy()

        bulk = pd.DataFrame(index=actionable.index)
        bulk["Product"] = "Sponsored Products"
        bulk["Entity"] = "Keyword"
        bulk["Operation"] = "Update"
        bulk["Campaign ID"] = actionable["campaign_id"]
        bulk["Ad Group ID"] = actionable["ad_group_id"]
        bulk["Keyword ID"] = actionable["keyword_id"]
        bulk["Campaign Name"] = actionable["campaign_name"]
        bulk["Ad Group Name"] = actionable["ad_group_name"]
        bulk["State"] = "Enabled"
        bulk["Keyword Text"] = actionable["target"]
        bulk["Match Type"] = actionable["match_type"]
        bulk["Bid"] = actionable["recommended_bid"]
        bulk["Daily Budget"] = ""
        bulk["Optimizer Action"] = actionable["recommended_action"]

        return bulk.reset_index(drop=True)

    def generate_harvest_bulk_updates(self, search_term_actions_df):
        actionable = search_term_actions_df[
            search_term_actions_df["search_term_action"] == "HARVEST_TO_EXACT"
        ].copy()

        actionable = actionable[
            actionable["campaign_id"].fillna("").astype(str).str.strip() != ""
        ]
        actionable = actionable[
            actionable["ad_group_id"].fillna("").astype(str).str.strip() != ""
        ]

        actionable["normalized_term"] = (
            actionable["search_term"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        actionable["ad_group_key"] = (
            actionable["campaign_name"].fillna("").astype(str).str.lower().str.strip()
            + "||"
            + actionable["ad_group_name"].fillna("").astype(str).str.lower().str.strip()
        )

        actionable = actionable[actionable["normalized_term"] != ""]
        actionable = actionable[actionable["ad_group_key"].isin(self.keyword_capable_ad_groups)]

        bulk = pd.DataFrame(index=actionable.index)
        bulk["Product"] = "Sponsored Products"
        bulk["Entity"] = "Keyword"
        bulk["Operation"] = "Create"
        bulk["Campaign ID"] = actionable["campaign_id"]
        bulk["Ad Group ID"] = actionable["ad_group_id"]
        bulk["Keyword ID"] = ""
        bulk["Campaign Name"] = actionable["campaign_name"]
        bulk["Ad Group Name"] = actionable["ad_group_name"]
        bulk["State"] = "Enabled"
        bulk["Keyword Text"] = actionable["normalized_term"]
        bulk["Match Type"] = "Exact"
        bulk["Bid"] = actionable["recommended_bid"]
        bulk["Daily Budget"] = ""
        bulk["Optimizer Action"] = actionable["search_term_action"]

        return bulk.reset_index(drop=True)

    def generate_negative_bulk_updates(self, search_term_actions_df):
        actionable = search_term_actions_df[
            search_term_actions_df["search_term_action"] == "ADD_NEGATIVE_PHRASE"
        ].copy()

        actionable = actionable[
            actionable["campaign_id"].fillna("").astype(str).str.strip() != ""
        ]
        actionable = actionable[
            actionable["ad_group_id"].fillna("").astype(str).str.strip() != ""
        ]

        actionable["normalized_term"] = (
            actionable["search_term"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        actionable = actionable[actionable["normalized_term"] != ""]

        actionable = actionable.drop_duplicates(
            subset=["campaign_id", "ad_group_id", "normalized_term"],
            keep="first",
        )

        bulk = pd.DataFrame(index=actionable.index)
        bulk["Product"] = "Sponsored Products"
        bulk["Entity"] = "Negative Keyword"
        bulk["Operation"] = "Create"
        bulk["Campaign ID"] = actionable["campaign_id"]
        bulk["Ad Group ID"] = actionable["ad_group_id"]
        bulk["Keyword ID"] = ""
        bulk["Campaign Name"] = actionable["campaign_name"]
        bulk["Ad Group Name"] = actionable["ad_group_name"]
        bulk["State"] = "Enabled"
        bulk["Keyword Text"] = actionable["normalized_term"]
        bulk["Match Type"] = "Negative Phrase"
        bulk["Bid"] = ""
        bulk["Daily Budget"] = ""
        bulk["Optimizer Action"] = actionable["search_term_action"]

        return bulk.reset_index(drop=True)

    def generate_budget_bulk_updates(self, campaign_budget_actions_df):
        actionable = campaign_budget_actions_df[
            campaign_budget_actions_df["campaign_action"] != "NO_ACTION"
        ].copy()

        bulk = pd.DataFrame(index=actionable.index)
        bulk["Product"] = "Sponsored Products"
        bulk["Entity"] = "Campaign"
        bulk["Operation"] = "Update"
        bulk["Campaign ID"] = actionable["campaign_id"]
        bulk["Ad Group ID"] = ""
        bulk["Keyword ID"] = ""
        bulk["Campaign Name"] = actionable["campaign_name"]
        bulk["Ad Group Name"] = ""
        bulk["State"] = "Enabled"
        bulk["Keyword Text"] = ""
        bulk["Match Type"] = ""
        bulk["Bid"] = ""
        bulk["Daily Budget"] = actionable["recommended_daily_budget"]
        bulk["Optimizer Action"] = actionable["campaign_action"]

        return bulk.reset_index(drop=True)

    # -----------------------------
    # SAFEGUARDS
    # -----------------------------
    def apply_final_safeguards(self, combined_bulk_updates):
        df = combined_bulk_updates.copy()

        required_columns = [
            "Product",
            "Entity",
            "Operation",
            "Campaign ID",
            "Ad Group ID",
            "Keyword ID",
            "Campaign Name",
            "Ad Group Name",
            "State",
            "Keyword Text",
            "Match Type",
            "Bid",
            "Daily Budget",
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        df = df[required_columns + [c for c in df.columns if c not in required_columns]]

        for col in [
            "Campaign ID",
            "Ad Group ID",
            "Keyword ID",
            "Campaign Name",
            "Ad Group Name",
            "Keyword Text",
            "Match Type",
            "Optimizer Action",
            "Entity",
            "Operation",
            "State",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
                df[col] = df[col].str.replace(r"\.0$", "", regex=True)

        df = df.drop_duplicates()

        signature_cols = [
            "Entity",
            "Campaign ID",
            "Ad Group ID",
            "Campaign Name",
            "Ad Group Name",
            "Keyword Text",
            "Match Type",
            "Operation",
        ]
        existing_signature_cols = [c for c in signature_cols if c in df.columns]

        if existing_signature_cols:
            df = df.drop_duplicates(subset=existing_signature_cols, keep="last")

        return df.reset_index(drop=True)

    # -----------------------------
    # SIMULATION SUMMARY
    # -----------------------------
    def build_simulation_summary(self, combined_bulk_updates, account_health):
        bid_increases = int((combined_bulk_updates["Optimizer Action"] == "INCREASE_BID").sum())
        bid_decreases = int((combined_bulk_updates["Optimizer Action"] == "DECREASE_BID").sum())
        negatives_added = int((combined_bulk_updates["Optimizer Action"] == "ADD_NEGATIVE_PHRASE").sum())
        harvested_keywords = int((combined_bulk_updates["Optimizer Action"] == "HARVEST_TO_EXACT").sum())
        budget_increases = int((combined_bulk_updates["Optimizer Action"] == "INCREASE_BUDGET").sum())
        budget_decreases = int((combined_bulk_updates["Optimizer Action"] == "DECREASE_BUDGET").sum())

        estimated_spend_impact_pct = round(
            (bid_increases * self.max_bid_up * 100)
            - (bid_decreases * self.max_bid_down * 100)
            + (budget_increases * self.budget_up_pct * 100)
            - (budget_decreases * self.budget_down_pct * 100),
            2,
        )

        return {
            "bid_increases": bid_increases,
            "bid_decreases": bid_decreases,
            "negatives_added": negatives_added,
            "harvested_keywords": harvested_keywords,
            "budget_increases": budget_increases,
            "budget_decreases": budget_decreases,
            "estimated_spend_impact_pct": estimated_spend_impact_pct,
            "account_roas": account_health["account_roas"],
            "waste_spend": account_health["waste_spend"],
            "waste_spend_pct": account_health["waste_spend_pct"],
            "health_status": account_health["health_status"],
            "adjusted_min_roas": account_health["adjusted_min_roas"],
            "tacos_pct": account_health["tacos_pct"],
            "tacos_status": account_health["tacos_status"],
        }

    # -----------------------------
    # PROCESS
    # -----------------------------
    def process(self):
        self.load_reports()

        search_terms = self.normalize_search_terms()
        targeting = self.normalize_targeting()
        impression_share = self.normalize_impression_share()
        bulk_targets = self.normalize_bulk_targets()
        bulk_campaigns = self.normalize_bulk_campaigns()

        targeting_with_share = self.join_impression_share_to_targeting(
            targeting,
            impression_share,
        )

        joined_targeting = self.join_targeting_to_bulk(
            targeting,
            bulk_targets,
        )

        account_health = self.build_account_health(targeting_with_share)
        adjusted_min_roas = account_health["adjusted_min_roas"]

        bid_recommendations = self.build_bid_recommendations(
            targeting_with_share,
            joined_targeting,
            adjusted_min_roas,
        )

        search_term_actions = self.build_search_term_actions(
            search_terms,
            adjusted_min_roas,
        )

        campaign_budget_actions = self.build_campaign_budget_actions(
            targeting_with_share,
            bulk_campaigns,
            adjusted_min_roas,
        )

        bid_bulk_updates = self.generate_bid_bulk_updates(bid_recommendations)
        harvest_bulk_updates = self.generate_harvest_bulk_updates(search_term_actions)
        negative_bulk_updates = self.generate_negative_bulk_updates(search_term_actions)
        budget_bulk_updates = self.generate_budget_bulk_updates(campaign_budget_actions)

        combined_bulk_updates = pd.concat(
            [
                bid_bulk_updates,
                harvest_bulk_updates,
                negative_bulk_updates,
                budget_bulk_updates,
            ],
            ignore_index=True,
        )

        combined_bulk_updates = self.apply_final_safeguards(combined_bulk_updates)
        simulation_summary = self.build_simulation_summary(combined_bulk_updates, account_health)

        campaign_health_dashboard = self.build_campaign_health_dashboard(
            targeting_with_share,
            adjusted_min_roas,
        )

        smart = self.build_smart_warnings(
            targeting_with_share_df=targeting_with_share,
            search_terms_df=search_terms,
            campaign_health_df=campaign_health_dashboard,
            account_health=account_health,
            adjusted_min_roas=adjusted_min_roas,
        )

        pre_run_preview = self.build_pre_run_preview(
            bid_recommendations=bid_recommendations,
            search_term_actions=search_term_actions,
            campaign_budget_actions=campaign_budget_actions,
        )

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
        }