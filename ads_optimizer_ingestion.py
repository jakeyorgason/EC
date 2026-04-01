import io
import os
import re
import calendar
from datetime import date

import numpy as np
import pandas as pd


class AdsOptimizerEngine:
    """
    Amazon Sponsored Products optimizer with:
    - bid updates
    - budget updates
    - keyword / ASIN graduation engine
    - automatic Dest / Research campaign creation
    - source cleanup via Negative Exact
    """

    ASIN_REGEX = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)

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
        max_bid_cap=5.0,
        max_budget_cap=500.0,
        enable_dest_graduation=True,
        enable_research_graduation=True,
        enable_asin_graduation=True,
        create_missing_dest_campaigns=True,
        create_missing_research_campaigns=True,
        dest_terms=("dest", "destination"),
        research_campaign_prefix="Research",
        min_orders_for_graduation=2,
        dest_acos_threshold=25.0,
        research_ctr_low=0.001,
        research_ctr_high=0.0025,
        research_cvr_low=0.02,
        research_cvr_high=0.05,
        loser_clicks_threshold=5,
        loser_ctr_threshold=0.0025,
        loser_cvr_threshold=0.05,
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
        self.zero_order_action = str(zero_order_action)
        self.strategy_mode = str(strategy_mode)
    
        self.enable_bid_updates = bool(enable_bid_updates)
        self.enable_search_harvesting = bool(enable_search_harvesting)
        self.enable_negative_keywords = bool(enable_negative_keywords)
        self.enable_budget_updates = bool(enable_budget_updates)
    
        self.enable_tacos_control = bool(enable_tacos_control)
        self.max_tacos_target = float(max_tacos_target) / 100.0
    
        self.enable_monthly_budget_control = bool(enable_monthly_budget_control)
        self.monthly_account_budget = float(monthly_account_budget)
        self.month_to_date_spend = float(month_to_date_spend)
        self.pacing_buffer_pct = float(pacing_buffer_pct) / 100.0
    
        self.max_bid_cap = float(max_bid_cap)
        self.max_budget_cap = float(max_budget_cap)
    
        self.enable_dest_graduation = bool(enable_dest_graduation)
        self.enable_research_graduation = bool(enable_research_graduation)
        self.enable_asin_graduation = bool(enable_asin_graduation)
        self.create_missing_dest_campaigns = bool(create_missing_dest_campaigns)
        self.create_missing_research_campaigns = bool(create_missing_research_campaigns)
    
        self.dest_terms = tuple(str(x).strip().lower() for x in dest_terms)
        self.research_campaign_prefix = str(research_campaign_prefix).strip() or "Research"
    
        self.min_orders_for_graduation = int(min_orders_for_graduation)
        self.dest_acos_threshold = float(dest_acos_threshold) / 100.0
        self.research_ctr_low = float(research_ctr_low)
        self.research_ctr_high = float(research_ctr_high)
        self.research_cvr_low = float(research_cvr_low)
        self.research_cvr_high = float(research_cvr_high)
        self.loser_clicks_threshold = int(loser_clicks_threshold)
        self.loser_ctr_threshold = float(loser_ctr_threshold)
        self.loser_cvr_threshold = float(loser_cvr_threshold)
        self.new_target_bid_multiplier = float(new_target_bid_multiplier)
        self.new_target_bid_cap = float(new_target_bid_cap)
    
        self.apply_strategy_settings()
    
        self.existing_exact_keywords = set()
        self.existing_negative_exact = set()
        self.existing_product_targets = set()
        self.keyword_capable_ad_groups = set()
    
        self.campaign_inventory = pd.DataFrame()
        self.ad_group_inventory = pd.DataFrame()
        self.bulk_targets_normalized = pd.DataFrame()
        self.bulk_campaigns_normalized = pd.DataFrame()

    def apply_strategy_settings(self):
        mode = self.strategy_mode.lower().strip()
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
            pos = None
            try:
                pos = file_obj.tell()
            except Exception:
                pass
            try:
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                data = file_obj.read()
            finally:
                try:
                    if pos is not None and hasattr(file_obj, "seek"):
                        file_obj.seek(pos)
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
        workbook = pd.read_excel(cloned, sheet_name=None, engine="openpyxl", dtype=str)
        if "Sponsored Products Campaigns" in workbook:
            return workbook["Sponsored Products Campaigns"]
        first_sheet = next(iter(workbook.keys()))
        return workbook[first_sheet]

    def load_sqp_simple_view(self):
        if self.sqp_report_file is None:
            return None
        cloned = self._clone_file_obj(self.sqp_report_file)
        return pd.read_csv(cloned, header=1)

    def safe_numeric(self, series):
        return pd.to_numeric(series, errors="coerce").fillna(0)

    def clean_text(self, series):
        cleaned = series.fillna("").astype(str).str.strip()
        cleaned = cleaned.replace("nan", "")
        cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
        return cleaned

    def blank_series(self, df):
        return pd.Series([""] * len(df), index=df.index)

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
        primary = self.get_optional_column(df, primary_candidates)
        fallback = self.get_optional_column(df, fallback_candidates)

        if primary is None and fallback is None:
            raise KeyError(
                f"Could not find a valid column for {label}. "
                f"Tried primary={primary_candidates}, fallback={fallback_candidates}."
            )

        if primary is not None:
            primary_series = self.clean_text(df[primary]).replace("", np.nan)
        else:
            primary_series = pd.Series([np.nan] * len(df), index=df.index)

        if fallback is not None:
            fallback_series = self.clean_text(df[fallback]).replace("", np.nan)
        else:
            fallback_series = pd.Series([np.nan] * len(df), index=df.index)

        return primary_series.combine_first(fallback_series).fillna("")

    def normalize_match_text(self, value):
        return " ".join(str(value or "").strip().lower().split())

    def normalize_term_text(self, value):
        text = str(value or "").lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def looks_like_asin(self, value):
        return bool(self.ASIN_REGEX.search(str(value or "").upper()))

    def extract_asin(self, value):
        match = self.ASIN_REGEX.search(str(value or "").upper())
        return match.group(0).upper() if match else ""

    def build_new_bid(self, cpc_value):
        bid = max(round(float(cpc_value or 0) * self.new_target_bid_multiplier, 2), 0.2)
        return round(min(bid, self.new_target_bid_cap), 2)

    def should_zero_order_negate(self):
        return self.zero_order_action in ["Add Negative", "Both"]

    def should_zero_order_decrease_bid(self):
        return self.zero_order_action in ["Decrease Bid", "Both"]

    def load_reports(self):
        self.bulk_df = self.load_bulk_sheet()
        self.search_df = self.load_file(self.search_term_file, expected_ext=".xlsx")
        self.targeting_df = self.load_file(self.targeting_file, expected_ext=".xlsx")
        self.impression_share_df = self.load_file(self.impression_share_file, expected_ext=".csv")
        self.business_df = self.load_file(self.business_report_file, expected_ext=".csv") if self.business_report_file is not None else None
        self.sqp_df = self.load_sqp_simple_view() if self.sqp_report_file is not None else None

    def calculate_metrics(self, df):
        df = df.copy()

        spend_col = self.get_optional_column(df, ["Spend"])
        clicks_col = self.get_optional_column(df, ["Clicks"])
        impressions_col = self.get_optional_column(df, ["Impressions"])
        orders_col = self.get_optional_column(df, ["7 Day Total Orders (#)"])
        sales_col = self.get_optional_column(df, ["7 Day Total Sales ", "7 Day Total Sales"])

        if spend_col:
            df[spend_col] = self.safe_numeric(df[spend_col])
        if clicks_col:
            df[clicks_col] = self.safe_numeric(df[clicks_col])
        if impressions_col:
            df[impressions_col] = self.safe_numeric(df[impressions_col])
        if orders_col:
            df[orders_col] = self.safe_numeric(df[orders_col])
        if sales_col:
            df[sales_col] = self.safe_numeric(df[sales_col])

        spend = self.safe_numeric(df[spend_col]) if spend_col else 0
        clicks = self.safe_numeric(df[clicks_col]) if clicks_col else 0
        impressions = self.safe_numeric(df[impressions_col]) if impressions_col else 0
        orders = self.safe_numeric(df[orders_col]) if orders_col else 0
        sales = self.safe_numeric(df[sales_col]) if sales_col else 0

        df["roas"] = np.where(spend > 0, sales / spend, 0)
        df["acos"] = np.where(sales > 0, spend / sales, 0)
        df["ctr"] = np.where(impressions > 0, clicks / impressions, 0)
        df["cvr"] = np.where(clicks > 0, orders / clicks, 0)
        df["cpc"] = np.where(clicks > 0, spend / clicks, 0)
        return df

    def normalize_search_terms(self):
        df = self.calculate_metrics(self.search_df.copy())
        normalized = pd.DataFrame()
        normalized["campaign_name"] = self.clean_text(df[self.get_first_existing_column(df, ["Campaign Name"], "campaign name")])
        normalized["ad_group_name"] = self.clean_text(df[self.get_first_existing_column(df, ["Ad Group Name"], "ad group name")])
        normalized["search_term"] = self.clean_text(df[self.get_first_existing_column(df, ["Customer Search Term"], "customer search term")])
        normalized["match_type"] = self.clean_text(df[self.get_first_existing_column(df, ["Match Type"], "match type")])

        normalized["clicks"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Clicks"], "clicks")])
        normalized["impressions"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Impressions"], "impressions")])
        normalized["spend"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Spend"], "spend")])
        normalized["orders"] = self.safe_numeric(df[self.get_first_existing_column(df, ["7 Day Total Orders (#)"], "orders")])

        sales_col = self.get_first_existing_column(df, ["7 Day Total Sales ", "7 Day Total Sales"], "sales")
        normalized["sales"] = self.safe_numeric(df[sales_col])

        normalized["roas"] = df["roas"]
        normalized["acos"] = df["acos"]
        normalized["ctr"] = df["ctr"]
        normalized["cvr"] = df["cvr"]
        normalized["cpc"] = df["cpc"]
        normalized["normalized_term"] = normalized["search_term"].map(self.normalize_term_text)
        normalized["is_asin"] = normalized["search_term"].map(self.looks_like_asin)
        normalized["asin_value"] = normalized["search_term"].map(self.extract_asin)

        return normalized

    def normalize_targeting(self):
        df = self.calculate_metrics(self.targeting_df.copy())
        normalized = pd.DataFrame()
        normalized["campaign_name"] = self.clean_text(df[self.get_first_existing_column(df, ["Campaign Name"], "campaign name")])
        normalized["ad_group_name"] = self.clean_text(df[self.get_first_existing_column(df, ["Ad Group Name"], "ad group name")])
        normalized["target"] = self.clean_text(df[self.get_first_existing_column(df, ["Targeting"], "targeting")])
        normalized["match_type"] = self.clean_text(df[self.get_first_existing_column(df, ["Match Type"], "match type")])

        normalized["clicks"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Clicks"], "clicks")])
        normalized["impressions"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Impressions"], "impressions")])
        normalized["spend"] = self.safe_numeric(df[self.get_first_existing_column(df, ["Spend"], "spend")])
        normalized["orders"] = self.safe_numeric(df[self.get_first_existing_column(df, ["7 Day Total Orders (#)"], "orders")])

        sales_col = self.get_first_existing_column(df, ["7 Day Total Sales ", "7 Day Total Sales"], "sales")
        normalized["sales"] = self.safe_numeric(df[sales_col])

        normalized["roas"] = df["roas"]
        normalized["acos"] = df["acos"]
        normalized["ctr"] = df["ctr"]
        normalized["cvr"] = df["cvr"]
        normalized["cpc"] = df["cpc"]
        return normalized

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

        normalized = pd.DataFrame()
        normalized["campaign_name"] = self.clean_text(df[campaign_col])
        normalized["ad_group_name"] = self.clean_text(df[ad_group_col])
        normalized["target"] = self.clean_text(df[target_col])
        normalized["match_type"] = self.clean_text(df[match_type_col])
        raw_share = (
            df[share_col].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
            .str.strip()
        )
        normalized["impression_share_pct"] = pd.to_numeric(raw_share, errors="coerce").fillna(0)
        return normalized

    def parse_campaign_name(self, campaign_name):
        raw = str(campaign_name or "").strip()
        parts = [p.strip() for p in raw.split("|")]
        parts += [""] * (4 - len(parts))
        description, advertised_item, campaign_type, suffix = parts[:4]

        desc_norm = self.normalize_match_text(description)
        is_dest = any(term in desc_norm for term in self.dest_terms)
        is_research = "research" in desc_norm
        parent_group = advertised_item.strip()

        return {
            "campaign_description": description,
            "advertised_item": advertised_item,
            "campaign_type": campaign_type,
            "suffix": suffix,
            "parent_group": parent_group,
            "is_dest": is_dest,
            "is_research": is_research,
            "description_norm": desc_norm,
        }

    def normalize_bulk_targets(self):
        df = self.bulk_df.copy()

        entity_col = self.get_first_existing_column(df, ["Entity"], "entity")
        match_type_col = self.get_optional_column(df, ["Match Type"])
        bid_col = self.get_optional_column(df, ["Bid"])
        keyword_col = self.get_optional_column(df, ["Keyword Text"])
        keyword_id_col = self.get_optional_column(df, ["Keyword ID"])
        campaign_id_col = self.get_optional_column(df, ["Campaign ID"])
        ad_group_id_col = self.get_optional_column(df, ["Ad Group ID"])
        portfolio_col = self.get_optional_column(df, ["Portfolio Name", "Portfolio Name (Informational only)"])

        product_expr_col = self.get_optional_column(
            df,
            [
                "Resolved Product Targeting Expression",
                "Product Targeting Expression",
                "Product Targeting Expression (Informational only)",
                "Resolved Product Targeting Expression (Informational only)",
            ],
        )

        rows = df[df[entity_col].isin(["Keyword", "Product Targeting", "Negative Keyword"])].copy()

        normalized = pd.DataFrame(index=rows.index)
        normalized["entity"] = self.clean_text(rows[entity_col])
        normalized["campaign_name"] = self.combine_preferred_columns(
            rows, ["Campaign Name"], ["Campaign Name (Informational only)"], "campaign name"
        )
        normalized["ad_group_name"] = self.combine_preferred_columns(
            rows, ["Ad Group Name"], ["Ad Group Name (Informational only)"], "ad group name"
        )
        normalized["portfolio_name"] = self.clean_text(rows[portfolio_col]) if portfolio_col else ""
        normalized["match_type"] = self.clean_text(rows[match_type_col]) if match_type_col else ""
        normalized["current_bid"] = self.safe_numeric(rows[bid_col]) if bid_col else 0.0
        normalized["keyword_id"] = self.clean_text(rows[keyword_id_col]) if keyword_id_col else ""
        normalized["campaign_id"] = self.clean_text(rows[campaign_id_col]) if campaign_id_col else ""
        normalized["ad_group_id"] = self.clean_text(rows[ad_group_id_col]) if ad_group_id_col else ""

        if keyword_col:
            keyword_text = self.clean_text(rows[keyword_col])
        else:
            keyword_text = self.blank_series(rows)

        if product_expr_col:
            product_expr = self.clean_text(rows[product_expr_col])
        else:
            product_expr = self.blank_series(rows)

        normalized["keyword_text"] = keyword_text
        normalized["product_expression"] = product_expr
        normalized["target"] = np.where(
            normalized["entity"].eq("Product Targeting"),
            normalized["product_expression"],
            normalized["keyword_text"],
        )
        normalized["target_norm"] = normalized["target"].map(self.normalize_term_text)

        parsed = normalized["campaign_name"].map(self.parse_campaign_name)
        normalized["campaign_description"] = parsed.map(lambda x: x["campaign_description"])
        normalized["advertised_item"] = parsed.map(lambda x: x["advertised_item"])
        normalized["campaign_type"] = parsed.map(lambda x: x["campaign_type"])
        normalized["suffix"] = parsed.map(lambda x: x["suffix"])
        normalized["parent_group"] = parsed.map(lambda x: x["parent_group"])
        normalized["is_dest"] = parsed.map(lambda x: x["is_dest"])
        normalized["is_research"] = parsed.map(lambda x: x["is_research"])

        existing_exact = normalized[
            (normalized["entity"] == "Keyword") & (normalized["match_type"].str.lower() == "exact")
        ].copy()
        self.existing_exact_keywords = set(
            (
                existing_exact["campaign_name"].map(self.normalize_match_text)
                + "||"
                + existing_exact["ad_group_name"].map(self.normalize_match_text)
                + "||"
                + existing_exact["keyword_text"].map(self.normalize_term_text)
            ).tolist()
        )

        existing_neg = normalized[
            (normalized["entity"] == "Negative Keyword") & (normalized["match_type"].str.lower() == "negative exact")
        ].copy()
        self.existing_negative_exact = set(
            (
                existing_neg["campaign_name"].map(self.normalize_match_text)
                + "||"
                + existing_neg["ad_group_name"].map(self.normalize_match_text)
                + "||"
                + existing_neg["keyword_text"].map(self.normalize_term_text)
            ).tolist()
        )

        existing_pat = normalized[normalized["entity"] == "Product Targeting"].copy()
        self.existing_product_targets = set(
            (
                existing_pat["campaign_name"].map(self.normalize_match_text)
                + "||"
                + existing_pat["ad_group_name"].map(self.normalize_match_text)
                + "||"
                + existing_pat["product_expression"].map(self.normalize_term_text)
            ).tolist()
        )

        keyword_rows = normalized[normalized["entity"] == "Keyword"].copy()
        self.keyword_capable_ad_groups = set(
            (
                keyword_rows["campaign_name"].map(self.normalize_match_text)
                + "||"
                + keyword_rows["ad_group_name"].map(self.normalize_match_text)
            ).tolist()
        )

        self.bulk_targets_normalized = normalized.reset_index(drop=True)
        return self.bulk_targets_normalized

    def normalize_bulk_campaigns(self):
        df = self.bulk_df.copy()

        entity_col = self.get_first_existing_column(df, ["Entity"], "entity")
        campaign_id_col = self.get_optional_column(df, ["Campaign ID"])
        budget_col = self.get_optional_column(df, ["Daily Budget"])
        state_col = self.get_optional_column(df, ["State"])
        portfolio_col = self.get_optional_column(df, ["Portfolio Name", "Portfolio Name (Informational only)"])

        rows = df[df[entity_col] == "Campaign"].copy()

        normalized = pd.DataFrame(index=rows.index)
        normalized["campaign_name"] = self.combine_preferred_columns(
            rows, ["Campaign Name"], ["Campaign Name (Informational only)"], "campaign name"
        )
        normalized["campaign_id"] = self.clean_text(rows[campaign_id_col]) if campaign_id_col else ""
        normalized["daily_budget"] = self.safe_numeric(rows[budget_col]) if budget_col else 0.0
        normalized["state"] = self.clean_text(rows[state_col]) if state_col else ""
        normalized["portfolio_name"] = self.clean_text(rows[portfolio_col]) if portfolio_col else ""

        parsed = normalized["campaign_name"].map(self.parse_campaign_name)
        normalized["campaign_description"] = parsed.map(lambda x: x["campaign_description"])
        normalized["advertised_item"] = parsed.map(lambda x: x["advertised_item"])
        normalized["campaign_type"] = parsed.map(lambda x: x["campaign_type"])
        normalized["suffix"] = parsed.map(lambda x: x["suffix"])
        normalized["parent_group"] = parsed.map(lambda x: x["parent_group"])
        normalized["is_dest"] = parsed.map(lambda x: x["is_dest"])
        normalized["is_research"] = parsed.map(lambda x: x["is_research"])

        self.bulk_campaigns_normalized = normalized.reset_index(drop=True)
        return self.bulk_campaigns_normalized

    def build_inventory(self):
        targets = self.bulk_targets_normalized.copy()
        campaigns = self.bulk_campaigns_normalized.copy()

        if campaigns.empty:
            self.campaign_inventory = pd.DataFrame()
            self.ad_group_inventory = pd.DataFrame()
            return

        ag = targets[targets["entity"].isin(["Keyword", "Product Targeting", "Negative Keyword"])].copy()
        if ag.empty:
            ad_groups = pd.DataFrame(columns=["campaign_name", "ad_group_name", "campaign_id", "ad_group_id", "portfolio_name"])
        else:
            ad_groups = (
                ag.groupby(["campaign_name", "ad_group_name"], as_index=False)
                .agg(
                    campaign_id=("campaign_id", "first"),
                    ad_group_id=("ad_group_id", "first"),
                    portfolio_name=("portfolio_name", "first"),
                )
            )

        ad_groups = ad_groups.merge(
            campaigns[
                [
                    "campaign_name",
                    "campaign_id",
                    "portfolio_name",
                    "advertised_item",
                    "parent_group",
                    "is_dest",
                    "is_research",
                    "campaign_type",
                    "daily_budget",
                ]
            ],
            on="campaign_name",
            how="left",
            suffixes=("", "_campaign"),
        )

        self.campaign_inventory = campaigns.copy()
        self.ad_group_inventory = ad_groups.copy()

    def join_impression_share_to_targeting(self, targeting_df, impression_share_df):
        t = targeting_df.copy()
        i = impression_share_df.copy()

        for df in [t, i]:
            df["_campaign_key"] = df["campaign_name"].map(self.normalize_match_text)
            df["_ad_group_key"] = df["ad_group_name"].map(self.normalize_match_text)
            df["_target_key"] = df["target"].map(self.normalize_term_text)
            df["_match_key"] = df["match_type"].map(self.normalize_match_text)

        joined = t.merge(
            i[["_campaign_key", "_ad_group_key", "_target_key", "_match_key", "impression_share_pct"]],
            on=["_campaign_key", "_ad_group_key", "_target_key", "_match_key"],
            how="left",
        )
        joined["impression_share_pct"] = joined["impression_share_pct"].fillna(0)
        return joined

    def join_targeting_to_bulk(self, targeting_df, bulk_targets_df):
        t = targeting_df.copy()
        b = bulk_targets_df.copy()

        t["_campaign_key"] = t["campaign_name"].map(self.normalize_match_text)
        t["_ad_group_key"] = t["ad_group_name"].map(self.normalize_match_text)
        t["_target_key"] = t["target"].map(self.normalize_term_text)
        t["_match_key"] = t["match_type"].map(self.normalize_match_text)

        b = b[b["entity"].isin(["Keyword", "Product Targeting"])].copy()
        b["_campaign_key"] = b["campaign_name"].map(self.normalize_match_text)
        b["_ad_group_key"] = b["ad_group_name"].map(self.normalize_match_text)
        b["_target_key"] = b["target"].map(self.normalize_term_text)
        b["_match_key"] = b["match_type"].map(self.normalize_match_text)

        joined = t.merge(
            b[
                [
                    "_campaign_key",
                    "_ad_group_key",
                    "_target_key",
                    "_match_key",
                    "campaign_id",
                    "ad_group_id",
                    "keyword_id",
                    "current_bid",
                    "entity",
                ]
            ],
            on=["_campaign_key", "_ad_group_key", "_target_key", "_match_key"],
            how="left",
        )
        return joined

    def build_business_sales_total(self):
        if self.business_df is None or not self.enable_tacos_control:
            return None

        df = self.business_df.copy()
        sales_col = self.get_optional_column(
            df,
            [
                "Ordered Product Sales",
                "Ordered Product Sales ",
                "Total Sales",
                "Sales",
                "Ordered Product Sales - B2C",
                "Ordered Product Sales - B2B",
            ],
        )
        if sales_col is None:
            return None

        sales_series = (
            df[sales_col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        total_sales = pd.to_numeric(sales_series, errors="coerce").fillna(0).sum()
        return float(total_sales)

    def build_account_health(self, targeting_with_share_df):
        df = targeting_with_share_df.copy()
        total_spend = float(df["spend"].sum())
        total_sales = float(df["sales"].sum())
        account_roas = round(total_sales / total_spend, 2) if total_spend > 0 else 0.0

        business_sales = self.build_business_sales_total()
        tacos_pct = None
        tacos_status = "not_enabled"
        if business_sales is not None and business_sales > 0:
            tacos = total_spend / business_sales
            tacos_pct = round(tacos * 100, 2)
            tacos_status = "above_target" if tacos > self.max_tacos_target else "within_target"

        adjusted_min_roas = self.min_roas
        health_status = "stable"

        if account_roas < self.min_roas:
            adjusted_min_roas = round(self.min_roas * self.account_health_tighten_multiplier, 2)
            health_status = "under_target"

        if tacos_status == "above_target":
            adjusted_min_roas = round(max(adjusted_min_roas, self.min_roas * 1.15), 2)
            health_status = "tacos_constrained"

        if account_roas >= self.min_roas * 1.2 and health_status == "stable":
            health_status = "above_target"

        return {
            "total_spend": round(total_spend, 2),
            "total_sales": round(total_sales, 2),
            "account_roas": account_roas,
            "adjusted_min_roas": adjusted_min_roas,
            "tacos_pct": tacos_pct,
            "tacos_status": tacos_status,
            "health_status": health_status,
        }

    def build_budget_pacing_status(self):
        if not self.enable_monthly_budget_control or self.monthly_account_budget <= 0:
            return {"enabled": False, "over_pace": False}

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
            "over_pace": bool(over_pace),
            "allowed_daily_pace": round(allowed_daily_pace, 2),
            "current_daily_pace": round(current_daily_pace, 2),
            "remaining_budget": round(remaining_budget, 2),
            "remaining_days": remaining_days,
        }

    def build_bid_recommendations(self, targeting_with_share_df, joined_targeting_df, adjusted_min_roas):
        recs = targeting_with_share_df.merge(
            joined_targeting_df[
                [
                    "campaign_name",
                    "ad_group_name",
                    "target",
                    "match_type",
                    "campaign_id",
                    "ad_group_id",
                    "keyword_id",
                    "current_bid",
                    "entity",
                ]
            ],
            on=["campaign_name", "ad_group_name", "target", "match_type"],
            how="left",
            suffixes=("", "_bulk"),
        )

        pacing = self.build_budget_pacing_status()
        over_pace = pacing["over_pace"]

        actions = []
        bids = []

        for _, row in recs.iterrows():
            current_bid = float(row.get("current_bid", 0) or 0)
            action = "NO_ACTION"
            new_bid = current_bid

            if current_bid <= 0 or not self.enable_bid_updates:
                actions.append(action)
                bids.append(new_bid)
                continue

            if (
                self.should_zero_order_decrease_bid()
                and row["clicks"] >= max(self.zero_order_click_threshold, 12)
                and row["orders"] == 0
            ):
                action = "DECREASE_BID"
                new_bid = round(max(current_bid * (1 - self.max_bid_down), 0.02), 2)

            elif row["roas"] < adjusted_min_roas and row["clicks"] >= self.min_clicks:
                action = "DECREASE_BID"
                new_bid = round(max(current_bid * (1 - self.max_bid_down), 0.02), 2)

            elif (
                not over_pace
                and row["roas"] > adjusted_min_roas * self.scale_roas_multiplier
                and row["orders"] >= self.min_orders_for_scaling
            ):
                action = "INCREASE_BID"
                new_bid = round(current_bid * (1 + self.max_bid_up), 2)

            new_bid = round(min(new_bid, self.max_bid_cap), 2)
            actions.append(action)
            bids.append(new_bid)

        recs["recommended_action"] = actions
        recs["recommended_bid"] = bids
        return recs

    def build_campaign_budget_actions(self, targeting_with_share_df, bulk_campaigns_df, adjusted_min_roas):
        campaign_perf = (
            targeting_with_share_df.groupby("campaign_name", as_index=False)
            .agg(
                clicks=("clicks", "sum"),
                orders=("orders", "sum"),
                spend=("spend", "sum"),
                sales=("sales", "sum"),
                avg_impression_share_pct=("impression_share_pct", "mean"),
            )
        )
        campaign_perf["roas"] = np.where(campaign_perf["spend"] > 0, campaign_perf["sales"] / campaign_perf["spend"], 0)

        recs = campaign_perf.merge(
            bulk_campaigns_df[["campaign_name", "campaign_id", "daily_budget"]],
            on="campaign_name",
            how="left",
        )

        pacing = self.build_budget_pacing_status()
        over_pace = pacing["over_pace"]

        actions = []
        budgets = []

        for _, row in recs.iterrows():
            action = "NO_ACTION"
            new_budget = float(row.get("daily_budget", 0) or 0)

            if new_budget <= 0 or not self.enable_budget_updates:
                actions.append(action)
                budgets.append(new_budget)
                continue

            if (
                not over_pace
                and row["roas"] >= adjusted_min_roas * 1.15
                and row["orders"] >= 3
                and row["avg_impression_share_pct"] < 20
            ):
                action = "INCREASE_BUDGET"
                new_budget = round(new_budget * (1 + self.budget_up_pct), 2)

            elif row["roas"] < adjusted_min_roas and row["clicks"] >= max(self.min_clicks * 3, 25):
                action = "DECREASE_BUDGET"
                new_budget = round(max(new_budget * (1 - self.budget_down_pct), 1.0), 2)

            new_budget = round(min(new_budget, self.max_budget_cap), 2)
            actions.append(action)
            budgets.append(new_budget)

        recs["campaign_action"] = actions
        recs["recommended_daily_budget"] = budgets
        return recs

    def find_source_campaign_meta(self, campaign_name):
        if self.campaign_inventory.empty:
            return None
        match = self.campaign_inventory[self.campaign_inventory["campaign_name"].eq(campaign_name)]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def find_existing_campaign(self, parent_group, portfolio_name, mode="dest"):
        if self.campaign_inventory.empty:
            return None

        df = self.campaign_inventory.copy()
        df = df[df["parent_group"].astype(str).str.strip() == str(parent_group).strip()]

        if str(portfolio_name or "").strip():
            df = df[df["portfolio_name"].astype(str).str.strip() == str(portfolio_name).strip()]

        if mode == "dest":
            df = df[df["is_dest"] == True]
            if df.empty:
                return None
            df = df.assign(
                dest_priority=np.where(
                    df["campaign_description"].astype(str).str.lower().str.contains("dest"),
                    0,
                    1,
                )
            )
            df = df.sort_values(by=["dest_priority", "campaign_name"])
            return df.iloc[0].to_dict()

        if mode == "research":
            df = df[df["is_research"] == True]
            if df.empty:
                return None
            return df.sort_values(by=["campaign_name"]).iloc[0].to_dict()

        return None

    def build_dest_campaign_name(self, parent_group, campaign_type="SP"):
        return f"Dest | {parent_group} | {campaign_type} | EC"

    def build_research_campaign_name(self, parent_group, campaign_type="SP"):
        return f"{self.research_campaign_prefix} | {parent_group} | {campaign_type} | EC"

    def keyword_exists_in_dest(self, campaign_name, ad_group_name, term):
        key = (
            self.normalize_match_text(campaign_name)
            + "||"
            + self.normalize_match_text(ad_group_name)
            + "||"
            + self.normalize_term_text(term)
        )
        return key in self.existing_exact_keywords

    def product_target_exists_in_dest(self, campaign_name, ad_group_name, product_expression):
        key = (
            self.normalize_match_text(campaign_name)
            + "||"
            + self.normalize_match_text(ad_group_name)
            + "||"
            + self.normalize_term_text(product_expression)
        )
        return key in self.existing_product_targets

    def loser_rule_hit(self, row):
        return (
            float(row.get("clicks", 0) or 0) > self.loser_clicks_threshold
            and float(row.get("ctr", 0) or 0) < self.loser_ctr_threshold
            and float(row.get("cvr", 0) or 0) < self.loser_cvr_threshold
        )

    def research_rule_hit(self, row):
        ctr = float(row.get("ctr", 0) or 0)
        cvr = float(row.get("cvr", 0) or 0)
        return (
            int(row.get("orders", 0) or 0) >= self.min_orders_for_graduation
            and (
                (self.research_ctr_low <= ctr <= self.research_ctr_high)
                or (self.research_cvr_low <= cvr <= self.research_cvr_high)
            )
        )

    def build_search_term_actions(self, search_terms_df, adjusted_min_roas):
        df = search_terms_df.copy()

        targeting_lookup = (
            self.bulk_targets_normalized[["campaign_name", "ad_group_name", "campaign_id", "ad_group_id"]]
            .drop_duplicates(subset=["campaign_name", "ad_group_name"])
        )

        df = df.merge(targeting_lookup, on=["campaign_name", "ad_group_name"], how="left")

        records = []

        for _, row in df.iterrows():
            record = row.to_dict()

            campaign_name = str(row["campaign_name"]).strip()
            normalized_term = str(row["normalized_term"]).strip()

            action = "NO_ACTION"
            reason = ""
            dest_campaign_name = ""
            dest_ad_group_name = ""
            source_negate = False
            create_campaign = False
            create_ad_group = False
            target_entity_type = ""
            target_match_type = ""
            product_expression = ""
            rec_bid = self.build_new_bid(row.get("cpc", 0))

            source_meta = self.find_source_campaign_meta(campaign_name)
            if not source_meta:
                record.update(
                    {
                        "search_term_action": action,
                        "reason": "Source campaign metadata not found in bulk sheet.",
                        "recommended_bid": rec_bid,
                    }
                )
                records.append(record)
                continue

            parent_group = str(source_meta.get("parent_group", "") or "").strip()
            portfolio_name = str(source_meta.get("portfolio_name", "") or "").strip()
            campaign_type = str(source_meta.get("campaign_type", "SP") or "SP").strip()

            if parent_group == "":
                if self.enable_negative_keywords and self.loser_rule_hit(row):
                    action = "ADD_NEGATIVE_EXACT"
                    reason = "Loser threshold met; source campaign has no routable parent group."
                record.update(
                    {
                        "search_term_action": action,
                        "reason": reason,
                        "recommended_bid": rec_bid,
                        "target_entity_type": target_entity_type,
                        "target_match_type": target_match_type,
                        "destination_campaign_name": dest_campaign_name,
                        "destination_ad_group_name": dest_ad_group_name,
                        "create_destination_campaign": create_campaign,
                        "create_destination_ad_group": create_ad_group,
                        "negate_source": source_negate,
                        "product_expression": product_expression,
                    }
                )
                records.append(record)
                continue

            if portfolio_name == "":
                if self.enable_negative_keywords and self.loser_rule_hit(row):
                    action = "ADD_NEGATIVE_EXACT"
                    reason = "Loser threshold met; source campaign has no portfolio so graduation was skipped."
                record.update(
                    {
                        "search_term_action": action,
                        "reason": reason,
                        "recommended_bid": rec_bid,
                        "target_entity_type": target_entity_type,
                        "target_match_type": target_match_type,
                        "destination_campaign_name": dest_campaign_name,
                        "destination_ad_group_name": dest_ad_group_name,
                        "create_destination_campaign": create_campaign,
                        "create_destination_ad_group": create_ad_group,
                        "negate_source": source_negate,
                        "product_expression": product_expression,
                    }
                )
                records.append(record)
                continue

            orders_ok = int(row["orders"]) >= self.min_orders_for_graduation
            dest_ok = orders_ok and float(row["acos"]) <= self.dest_acos_threshold

            if self.enable_negative_keywords and self.loser_rule_hit(row):
                action = "ADD_NEGATIVE_EXACT"
                reason = "Loser threshold met."

            elif row["is_asin"] and orders_ok and dest_ok and self.enable_asin_graduation:
                existing_dest = self.find_existing_campaign(parent_group, portfolio_name, mode="dest")
                if existing_dest is None and self.create_missing_dest_campaigns:
                    dest_campaign_name = self.build_dest_campaign_name(parent_group, campaign_type)
                    create_campaign = True
                elif existing_dest is not None:
                    dest_campaign_name = existing_dest["campaign_name"]

                dest_ad_group_name = "ASIN Targets"
                product_expression = f'asin="{row["asin_value"]}"'
                target_entity_type = "Product Targeting"
                target_match_type = "Targeting Expression"

                if dest_campaign_name:
                    exists = self.product_target_exists_in_dest(dest_campaign_name, dest_ad_group_name, product_expression)
                    if exists:
                        action = "NEGATE_SOURCE_EXISTING_DEST"
                        reason = "ASIN already exists in Dest; source cleanup only."
                        source_negate = True
                    else:
                        action = "ADD_ASIN_TO_DEST"
                        reason = "ASIN winner graduated to Dest."
                        source_negate = True
                        create_ad_group = True
                else:
                    action = "NO_ACTION"
                    reason = "No valid Dest campaign found and creation disabled."

            elif dest_ok and self.enable_dest_graduation:
                existing_dest = self.find_existing_campaign(parent_group, portfolio_name, mode="dest")
                if existing_dest is None and self.create_missing_dest_campaigns:
                    dest_campaign_name = self.build_dest_campaign_name(parent_group, campaign_type)
                    create_campaign = True
                elif existing_dest is not None:
                    dest_campaign_name = existing_dest["campaign_name"]

                dest_ad_group_name = "Graduated Keywords"
                target_entity_type = "Keyword"
                target_match_type = "Exact"

                if dest_campaign_name:
                    exists = self.keyword_exists_in_dest(dest_campaign_name, dest_ad_group_name, normalized_term)
                    if exists:
                        action = "NEGATE_SOURCE_EXISTING_DEST"
                        reason = "Keyword already exists in Dest; source cleanup only."
                        source_negate = True
                    else:
                        action = "ADD_TO_DEST_EXACT"
                        reason = "Winner graduated to Dest Exact."
                        source_negate = True
                        create_ad_group = True
                else:
                    action = "NO_ACTION"
                    reason = "No valid Dest campaign found and creation disabled."

            elif self.enable_research_graduation and self.research_rule_hit(row):
                existing_research = self.find_existing_campaign(parent_group, portfolio_name, mode="research")
                if existing_research is None and self.create_missing_research_campaigns:
                    dest_campaign_name = self.build_research_campaign_name(parent_group, campaign_type)
                    create_campaign = True
                elif existing_research is not None:
                    dest_campaign_name = existing_research["campaign_name"]

                dest_ad_group_name = "Phrase Research"
                target_entity_type = "Keyword"
                target_match_type = "Phrase"

                if dest_campaign_name:
                    action = "ADD_TO_RESEARCH_PHRASE"
                    reason = "Converting term routed to Research Phrase."
                    source_negate = True
                    create_ad_group = True
                else:
                    action = "NO_ACTION"
                    reason = "No valid Research campaign found and creation disabled."

            record.update(
                {
                    "search_term_action": action,
                    "reason": reason,
                    "recommended_bid": rec_bid,
                    "target_entity_type": target_entity_type,
                    "target_match_type": target_match_type,
                    "destination_campaign_name": dest_campaign_name,
                    "destination_ad_group_name": dest_ad_group_name,
                    "create_destination_campaign": create_campaign,
                    "create_destination_ad_group": create_ad_group,
                    "negate_source": source_negate,
                    "product_expression": product_expression,
                    "source_parent_group": parent_group,
                    "source_portfolio_name": portfolio_name,
                    "source_campaign_type": campaign_type,
                }
            )
            records.append(record)

        return pd.DataFrame(records)

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

        campaign_health["roas"] = np.where(campaign_health["spend"] > 0, campaign_health["sales"] / campaign_health["spend"], 0)
        campaign_health["acos"] = np.where(campaign_health["sales"] > 0, campaign_health["spend"] / campaign_health["sales"], 0)

        conditions = [
            (campaign_health["spend"] >= 100) & (campaign_health["orders"] == 0),
            (campaign_health["roas"] < adjusted_min_roas) & (campaign_health["clicks"] >= self.min_clicks),
            (
                (campaign_health["roas"] >= adjusted_min_roas * 1.15)
                & (campaign_health["orders"] >= 3)
                & (campaign_health["avg_impression_share_pct"] < 20)
            ),
        ]
        choices = ["Waste Alert", "Under Target", "Scalable"]

        campaign_health["campaign_status"] = np.select(conditions, choices, default="Stable")
        campaign_health["avg_impression_share_pct"] = campaign_health["avg_impression_share_pct"].round(2)
        campaign_health["roas"] = campaign_health["roas"].round(2)
        campaign_health["acos"] = (campaign_health["acos"] * 100).round(2)

        return campaign_health.sort_values(by=["spend", "sales"], ascending=[False, False]).reset_index(drop=True)

    def normalize_sqp(self):
        if self.sqp_df is None or self.sqp_df.empty:
            return pd.DataFrame()

        df = self.sqp_df.copy()
        normalized = pd.DataFrame()
        normalized["search_query"] = self.clean_text(df["Search Query"])
        normalized["search_query_score"] = self.safe_numeric(df["Search Query Score"])
        normalized["search_query_volume"] = self.safe_numeric(df["Search Query Volume"])
        normalized["purchases_total_count"] = self.safe_numeric(df["Purchases: Total Count"])
        normalized["purchase_rate_pct"] = self.safe_numeric(
            df["Purchases: Purchase Rate %"].astype(str).str.replace("%", "", regex=False)
        )
        normalized["purchases_brand_share_pct"] = self.safe_numeric(
            df["Purchases: Brand Share %"].astype(str).str.replace("%", "", regex=False)
        )
        normalized = normalized[normalized["search_query"] != ""].copy()
        return normalized.reset_index(drop=True)

    def build_sqp_opportunities(self, sqp_df, search_terms_df):
        if sqp_df is None or sqp_df.empty:
            return pd.DataFrame(), {
                "uploaded": False,
                "total_queries": 0,
                "high_opportunity": 0,
                "monitor": 0,
                "low_priority": 0,
                "graduation_overlap": 0,
            }

        search_term_set = set(search_terms_df["normalized_term"].fillna("").astype(str).tolist()) if not search_terms_df.empty else set()
        sqp = sqp_df.copy()
        sqp["search_query_key"] = sqp["search_query"].map(self.normalize_term_text)
        sqp["in_search_term_report"] = sqp["search_query_key"].isin(search_term_set)

        sqp["opportunity_tier"] = np.select(
            [
                (
                    (sqp["search_query_volume"] >= 1000)
                    & (sqp["purchase_rate_pct"] >= 6)
                    & (sqp["purchases_total_count"] >= 25)
                    & (sqp["purchases_brand_share_pct"] < 20)
                    & (sqp["search_query_score"] >= 300)
                ),
                (
                    (sqp["search_query_volume"] >= 500)
                    & (sqp["purchase_rate_pct"] >= 4)
                    & (sqp["purchases_total_count"] >= 10)
                    & (sqp["purchases_brand_share_pct"] < 30)
                ),
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
            ["Test as Exact / Phrase", "Prioritize current search term for graduation", "Monitor and test selectively"],
            default="No immediate action",
        )

        summary = {
            "uploaded": True,
            "total_queries": int(len(sqp)),
            "high_opportunity": int((sqp["opportunity_tier"] == "High Opportunity").sum()),
            "monitor": int((sqp["opportunity_tier"] == "Monitor").sum()),
            "low_priority": int((sqp["opportunity_tier"] == "Low Priority").sum()),
            "graduation_overlap": int(((sqp["opportunity_tier"] == "High Opportunity") & (sqp["in_search_term_report"])).sum()),
        }
        return sqp.reset_index(drop=True), summary

    def build_smart_warnings(self, targeting_with_share_df, search_terms_df, campaign_health_df, account_health, adjusted_min_roas, sqp_summary=None):
        warnings = []
        suggestions = []

        waste_campaigns = campaign_health_df[(campaign_health_df["spend"] >= 100) & (campaign_health_df["orders"] == 0)]
        scalable_campaigns = campaign_health_df[
            (campaign_health_df["roas"] >= adjusted_min_roas * 1.15)
            & (campaign_health_df["orders"] >= 3)
            & (campaign_health_df["avg_impression_share_pct"] < 20)
        ]
        graduation_candidates = search_terms_df[
            search_terms_df["search_term_action"].isin(["ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST"])
        ]
        loser_terms = search_terms_df[search_terms_df["search_term_action"] == "ADD_NEGATIVE_EXACT"]

        if len(waste_campaigns) > 0:
            warnings.append(f"{len(waste_campaigns)} campaign(s) have at least $100 spend and 0 orders.")
            suggestions.append("Review waste campaigns for tighter bids and negatives.")

        if len(scalable_campaigns) > 0:
            warnings.append(f"{len(scalable_campaigns)} campaign(s) have strong ROAS with low impression share.")
            suggestions.append("These campaigns may support careful bid or budget scaling.")

        if len(graduation_candidates) > 0:
            warnings.append(f"{len(graduation_candidates)} search term(s) are ready for graduation or routing.")
            suggestions.append("Review Dest / Research campaign routing and source cleanup.")

        if len(loser_terms) > 0:
            warnings.append(f"{len(loser_terms)} search term(s) qualify for Negative Exact cleanup.")
            suggestions.append("Confirm loser negatives align with your account structure.")

        if account_health.get("health_status") == "under_target":
            warnings.append(f"Account ROAS is below target at {account_health.get('account_roas')}.")
            suggestions.append("Keep scaling selective and prioritize efficiency.")

        if account_health.get("tacos_status") == "above_target":
            warnings.append("Account TACOS is above the configured guardrail.")
            suggestions.append("Tighten scaling and prioritize profitable terms.")

        pacing = self.build_budget_pacing_status()
        if pacing.get("enabled") and pacing.get("over_pace"):
            warnings.append("Monthly budget pacing is currently over target.")
            suggestions.append("Suppress scaling until monthly pace returns to target.")

        if sqp_summary and sqp_summary.get("uploaded") and sqp_summary.get("high_opportunity", 0) > 0:
            warnings.append(f"{sqp_summary['high_opportunity']} high-opportunity SQP queries were identified.")
            suggestions.append("Use SQP as context for future keyword expansion.")

        if not warnings:
            warnings.append("No major risk flags detected from the uploaded reports.")
        if not suggestions:
            suggestions.append("Account looks stable under the current ruleset.")

        return {"warnings": warnings[:8], "suggestions": suggestions[:8]}

    def generate_bid_bulk_updates(self, recommendations_df):
        actionable = recommendations_df[
            (recommendations_df["recommended_action"] != "NO_ACTION") & (recommendations_df["current_bid"] > 0)
        ].copy()

        if actionable.empty:
            return pd.DataFrame()

        bulk = pd.DataFrame(index=actionable.index)
        bulk["Product"] = "Sponsored Products"
        bulk["Entity"] = "Keyword"
        bulk["Operation"] = "Update"
        bulk["Campaign ID"] = actionable["campaign_id"]
        bulk["Ad Group ID"] = actionable["ad_group_id"]
        bulk["Keyword ID"] = actionable["keyword_id"]
        bulk["Campaign Name"] = actionable["campaign_name"]
        bulk["Ad Group Name"] = actionable["ad_group_name"]
        bulk["Portfolio Name"] = ""
        bulk["State"] = "Enabled"
        bulk["Keyword Text"] = actionable["target"]
        bulk["Match Type"] = actionable["match_type"]
        bulk["Bid"] = actionable["recommended_bid"]
        bulk["Daily Budget"] = ""
        bulk["Product Targeting Expression"] = ""
        bulk["Optimizer Action"] = actionable["recommended_action"]
        return bulk.reset_index(drop=True)

    def generate_budget_bulk_updates(self, campaign_budget_actions_df):
        actionable = campaign_budget_actions_df[campaign_budget_actions_df["campaign_action"] != "NO_ACTION"].copy()
        if actionable.empty:
            return pd.DataFrame()

        bulk = pd.DataFrame(index=actionable.index)
        bulk["Product"] = "Sponsored Products"
        bulk["Entity"] = "Campaign"
        bulk["Operation"] = "Update"
        bulk["Campaign ID"] = actionable["campaign_id"]
        bulk["Ad Group ID"] = ""
        bulk["Keyword ID"] = ""
        bulk["Campaign Name"] = actionable["campaign_name"]
        bulk["Ad Group Name"] = ""
        bulk["Portfolio Name"] = ""
        bulk["State"] = "Enabled"
        bulk["Keyword Text"] = ""
        bulk["Match Type"] = ""
        bulk["Bid"] = ""
        bulk["Daily Budget"] = actionable["recommended_daily_budget"]
        bulk["Product Targeting Expression"] = ""
        bulk["Optimizer Action"] = actionable["campaign_action"]
        return bulk.reset_index(drop=True)

    def generate_campaign_create_bulk_updates(self, search_term_actions_df):
        actions = search_term_actions_df[
            search_term_actions_df["search_term_action"].isin(["ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST"])
            & (search_term_actions_df["create_destination_campaign"] == True)
        ].copy()
    
        if actions.empty:
            return pd.DataFrame()
    
        rows = []
        seen = set()
    
        for _, row in actions.iterrows():
            campaign_name = str(row.get("destination_campaign_name", "")).strip()
            if campaign_name == "" or campaign_name in seen:
                continue
            seen.add(campaign_name)
    
            source_portfolio = str(row.get("source_portfolio_name", "") or "").strip()
            source_campaign_type = str(row.get("source_campaign_type", "SP") or "SP").strip()
    
            rows.append(
                {
                    "Product": "Sponsored Products",
                    "Entity": "Campaign",
                    "Operation": "Create",
                    "Campaign ID": "",
                    "Ad Group ID": "",
                    "Keyword ID": "",
                    "Campaign Name": campaign_name,
                    "Ad Group Name": "",
                    "Portfolio Name": source_portfolio,
                    "State": "Enabled",
                    "Targeting Type": "Manual",
                    "Campaign Type": source_campaign_type,
                    "Keyword Text": "",
                    "Match Type": "",
                    "Bid": "",
                    "Daily Budget": 10.0,
                    "Product Targeting Expression": "",
                    "Optimizer Action": "CREATE_CAMPAIGN",
                }
            )
    
        return pd.DataFrame(rows)

    def generate_ad_group_create_bulk_updates(self, search_term_actions_df):
        actions = search_term_actions_df[
            search_term_actions_df["search_term_action"].isin(["ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST"])
        ].copy()
    
        if actions.empty:
            return pd.DataFrame()
    
        rows = []
        seen = set()
    
        for _, row in actions.iterrows():
            campaign_name = str(row.get("destination_campaign_name", "")).strip()
            ad_group_name = str(row.get("destination_ad_group_name", "")).strip()
    
            if campaign_name == "" or ad_group_name == "":
                continue
    
            ad_group_key = self.normalize_match_text(campaign_name) + "||" + self.normalize_match_text(ad_group_name)
            if ad_group_key in seen:
                continue
    
            campaign_id = ""
            if not self.campaign_inventory.empty:
                existing_campaign = self.campaign_inventory[
                    self.campaign_inventory["campaign_name"].map(self.normalize_match_text)
                    == self.normalize_match_text(campaign_name)
                ]
                if not existing_campaign.empty:
                    campaign_id = str(existing_campaign.iloc[0].get("campaign_id", "") or "").strip()
    
            if not self.ad_group_inventory.empty:
                existing_ad_group = self.ad_group_inventory[
                    (self.ad_group_inventory["campaign_name"].map(self.normalize_match_text) == self.normalize_match_text(campaign_name))
                    & (self.ad_group_inventory["ad_group_name"].map(self.normalize_match_text) == self.normalize_match_text(ad_group_name))
                ]
                if not existing_ad_group.empty:
                    continue
    
            seen.add(ad_group_key)
    
            rows.append(
                {
                    "Product": "Sponsored Products",
                    "Entity": "Ad Group",
                    "Operation": "Create",
                    "Campaign ID": campaign_id,
                    "Ad Group ID": "",
                    "Keyword ID": "",
                    "Campaign Name": campaign_name,
                    "Ad Group Name": ad_group_name,
                    "Portfolio Name": str(row.get("source_portfolio_name", "") or "").strip(),
                    "State": "Enabled",
                    "Targeting Type": "",
                    "Campaign Type": "",
                    "Keyword Text": "",
                    "Match Type": "",
                    "Bid": round(float(row.get("recommended_bid", 0.5) or 0.5), 2),
                    "Daily Budget": "",
                    "Product Targeting Expression": "",
                    "Optimizer Action": "CREATE_AD_GROUP",
                }
            )
    
        return pd.DataFrame(rows)

    def generate_keyword_graduation_bulk_updates(self, search_term_actions_df):
        actionable = search_term_actions_df[
            search_term_actions_df["search_term_action"].isin(["ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE"])
        ].copy()
    
        if actionable.empty:
            return pd.DataFrame()
    
        rows = []
    
        for _, row in actionable.iterrows():
            campaign_name = str(row.get("destination_campaign_name", "")).strip()
            ad_group_name = str(row.get("destination_ad_group_name", "")).strip()
    
            campaign_id = ""
            ad_group_id = ""
    
            if not self.campaign_inventory.empty:
                campaign_match = self.campaign_inventory[
                    self.campaign_inventory["campaign_name"].map(self.normalize_match_text)
                    == self.normalize_match_text(campaign_name)
                ]
                if not campaign_match.empty:
                    campaign_id = str(campaign_match.iloc[0].get("campaign_id", "") or "").strip()
    
            if not self.ad_group_inventory.empty:
                ad_group_match = self.ad_group_inventory[
                    (self.ad_group_inventory["campaign_name"].map(self.normalize_match_text) == self.normalize_match_text(campaign_name))
                    & (self.ad_group_inventory["ad_group_name"].map(self.normalize_match_text) == self.normalize_match_text(ad_group_name))
                ]
                if not ad_group_match.empty:
                    ad_group_id = str(ad_group_match.iloc[0].get("ad_group_id", "") or "").strip()
    
            rows.append(
                {
                    "Product": "Sponsored Products",
                    "Entity": "Keyword",
                    "Operation": "Create",
                    "Campaign ID": campaign_id,
                    "Ad Group ID": ad_group_id,
                    "Keyword ID": "",
                    "Campaign Name": campaign_name,
                    "Ad Group Name": ad_group_name,
                    "Portfolio Name": str(row.get("source_portfolio_name", "") or "").strip(),
                    "State": "Enabled",
                    "Targeting Type": "",
                    "Campaign Type": "",
                    "Keyword Text": row["normalized_term"],
                    "Match Type": row["target_match_type"],
                    "Bid": round(float(row["recommended_bid"]), 2),
                    "Daily Budget": "",
                    "Product Targeting Expression": "",
                    "Optimizer Action": row["search_term_action"],
                }
            )
    
        return pd.DataFrame(rows)

    def generate_product_target_bulk_updates(self, search_term_actions_df):
        actionable = search_term_actions_df[search_term_actions_df["search_term_action"] == "ADD_ASIN_TO_DEST"].copy()
    
        if actionable.empty:
            return pd.DataFrame()
    
        rows = []
    
        for _, row in actionable.iterrows():
            campaign_name = str(row.get("destination_campaign_name", "")).strip()
            ad_group_name = str(row.get("destination_ad_group_name", "")).strip()
    
            campaign_id = ""
            ad_group_id = ""
    
            if not self.campaign_inventory.empty:
                campaign_match = self.campaign_inventory[
                    self.campaign_inventory["campaign_name"].map(self.normalize_match_text)
                    == self.normalize_match_text(campaign_name)
                ]
                if not campaign_match.empty:
                    campaign_id = str(campaign_match.iloc[0].get("campaign_id", "") or "").strip()
    
            if not self.ad_group_inventory.empty:
                ad_group_match = self.ad_group_inventory[
                    (self.ad_group_inventory["campaign_name"].map(self.normalize_match_text) == self.normalize_match_text(campaign_name))
                    & (self.ad_group_inventory["ad_group_name"].map(self.normalize_match_text) == self.normalize_match_text(ad_group_name))
                ]
                if not ad_group_match.empty:
                    ad_group_id = str(ad_group_match.iloc[0].get("ad_group_id", "") or "").strip()
    
            rows.append(
                {
                    "Product": "Sponsored Products",
                    "Entity": "Product Targeting",
                    "Operation": "Create",
                    "Campaign ID": campaign_id,
                    "Ad Group ID": ad_group_id,
                    "Keyword ID": "",
                    "Campaign Name": campaign_name,
                    "Ad Group Name": ad_group_name,
                    "Portfolio Name": str(row.get("source_portfolio_name", "") or "").strip(),
                    "State": "Enabled",
                    "Targeting Type": "",
                    "Campaign Type": "",
                    "Keyword Text": "",
                    "Match Type": "",
                    "Bid": round(float(row["recommended_bid"]), 2),
                    "Daily Budget": "",
                    "Product Targeting Expression": row["product_expression"],
                    "Optimizer Action": row["search_term_action"],
                }
            )
    
        return pd.DataFrame(rows)

    def generate_negative_bulk_updates(self, search_term_actions_df):
        actionable = search_term_actions_df[
            (search_term_actions_df["search_term_action"] == "ADD_NEGATIVE_EXACT")
            | (search_term_actions_df["negate_source"] == True)
            | (search_term_actions_df["search_term_action"] == "NEGATE_SOURCE_EXISTING_DEST")
        ].copy()

        actionable = actionable[actionable["campaign_id"].fillna("").astype(str).str.strip() != ""]
        actionable = actionable[actionable["ad_group_id"].fillna("").astype(str).str.strip() != ""]

        if actionable.empty:
            return pd.DataFrame()

        rows = []
        seen = set()

        for _, row in actionable.iterrows():
            neg_term = row["asin_value"] if row.get("is_asin", False) else row["normalized_term"]
            neg_term = str(neg_term or "").strip()
            if neg_term == "":
                continue

            source_key = (
                self.normalize_match_text(row["campaign_name"])
                + "||"
                + self.normalize_match_text(row["ad_group_name"])
                + "||"
                + self.normalize_term_text(neg_term)
            )
            if source_key in seen or source_key in self.existing_negative_exact:
                continue

            seen.add(source_key)
            rows.append(
                {
                    "Product": "Sponsored Products",
                    "Entity": "Negative Keyword",
                    "Operation": "Create",
                    "Campaign ID": row["campaign_id"],
                    "Ad Group ID": row["ad_group_id"],
                    "Keyword ID": "",
                    "Campaign Name": row["campaign_name"],
                    "Ad Group Name": row["ad_group_name"],
                    "Portfolio Name": row.get("source_portfolio_name", ""),
                    "State": "Enabled",
                    "Keyword Text": neg_term,
                    "Match Type": "Negative Exact",
                    "Bid": "",
                    "Daily Budget": "",
                    "Product Targeting Expression": "",
                    "Optimizer Action": "ADD_NEGATIVE_EXACT",
                }
            )

        return pd.DataFrame(rows)

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
            "Portfolio Name",
            "State",
            "Targeting Type",
            "Campaign Type",
            "Keyword Text",
            "Match Type",
            "Bid",
            "Daily Budget",
            "Product Targeting Expression",
            "Optimizer Action",
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        df = df[required_columns + [c for c in df.columns if c not in required_columns]]

        text_cols = [
            "Campaign ID",
            "Ad Group ID",
            "Keyword ID",
            "Campaign Name",
            "Ad Group Name",
            "Portfolio Name",
            "Keyword Text",
            "Match Type",
            "Product Targeting Expression",
            "Optimizer Action",
            "Entity",
            "Operation",
            "State",
        ]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
            df[col] = df[col].str.replace(r"\.0$", "", regex=True)

        df = df.drop_duplicates()

        sig_cols = [
            "Entity",
            "Operation",
            "Campaign Name",
            "Ad Group Name",
            "Keyword Text",
            "Match Type",
            "Product Targeting Expression",
            "Optimizer Action",
        ]
        df = df.drop_duplicates(subset=sig_cols, keep="last")

        return df.reset_index(drop=True)

    def build_pre_run_preview(self, bid_recommendations, search_term_actions, campaign_budget_actions):
        return {
            "bid_increases": int((bid_recommendations["recommended_action"] == "INCREASE_BID").sum()),
            "bid_decreases": int((bid_recommendations["recommended_action"] == "DECREASE_BID").sum()),
            "negatives_added": int(
                search_term_actions["search_term_action"].isin(["ADD_NEGATIVE_EXACT", "NEGATE_SOURCE_EXISTING_DEST"]).sum()
                + search_term_actions["negate_source"].fillna(False).sum()
            ),
            "graduations": int(
                search_term_actions["search_term_action"].isin(["ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST"]).sum()
            ),
            "dest_exact": int((search_term_actions["search_term_action"] == "ADD_TO_DEST_EXACT").sum()),
            "research_phrase": int((search_term_actions["search_term_action"] == "ADD_TO_RESEARCH_PHRASE").sum()),
            "asin_dest": int((search_term_actions["search_term_action"] == "ADD_ASIN_TO_DEST").sum()),
            "campaign_creates": int(search_term_actions["create_destination_campaign"].fillna(False).sum()),
            "budget_increases": int((campaign_budget_actions["campaign_action"] == "INCREASE_BUDGET").sum()),
            "budget_decreases": int((campaign_budget_actions["campaign_action"] == "DECREASE_BUDGET").sum()),
        }

    def build_simulation_summary(self, combined_bulk_updates, account_health):
        df = combined_bulk_updates.copy()
        return {
            "bid_increases": int((df["Optimizer Action"] == "INCREASE_BID").sum()),
            "bid_decreases": int((df["Optimizer Action"] == "DECREASE_BID").sum()),
            "negatives_added": int((df["Optimizer Action"] == "ADD_NEGATIVE_EXACT").sum()),
            "graduations": int(df["Optimizer Action"].isin(["ADD_TO_DEST_EXACT", "ADD_TO_RESEARCH_PHRASE", "ADD_ASIN_TO_DEST"]).sum()),
            "dest_exact": int((df["Optimizer Action"] == "ADD_TO_DEST_EXACT").sum()),
            "research_phrase": int((df["Optimizer Action"] == "ADD_TO_RESEARCH_PHRASE").sum()),
            "asin_dest": int((df["Optimizer Action"] == "ADD_ASIN_TO_DEST").sum()),
            "campaign_creates": int((df["Optimizer Action"] == "CREATE_CAMPAIGN").sum()),
            "ad_group_creates": int((df["Optimizer Action"] == "CREATE_AD_GROUP").sum()),
            "budget_increases": int((df["Optimizer Action"] == "INCREASE_BUDGET").sum()),
            "budget_decreases": int((df["Optimizer Action"] == "DECREASE_BUDGET").sum()),
            "estimated_spend_impact_pct": 0.0,
            "account_roas": account_health.get("account_roas"),
            "tacos_pct": account_health.get("tacos_pct"),
        }

    def save_run_history(self, simulation_summary, account_health):
        history_path = "run_history.csv"
        row = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy_mode": self.strategy_mode,
            "min_roas": self.min_roas,
            "min_clicks": self.min_clicks,
            "account_roas": account_health.get("account_roas"),
            "tacos_pct": account_health.get("tacos_pct"),
            "health_status": account_health.get("health_status"),
            "bid_increases": simulation_summary.get("bid_increases"),
            "bid_decreases": simulation_summary.get("bid_decreases"),
            "negatives_added": simulation_summary.get("negatives_added"),
            "graduations": simulation_summary.get("graduations"),
            "campaign_creates": simulation_summary.get("campaign_creates"),
            "budget_increases": simulation_summary.get("budget_increases"),
            "budget_decreases": simulation_summary.get("budget_decreases"),
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

    def analyze(self):
        self.load_reports()

        search_terms = self.normalize_search_terms()
        targeting = self.normalize_targeting()
        impression_share = self.normalize_impression_share()
        self.normalize_bulk_targets()
        self.normalize_bulk_campaigns()
        self.build_inventory()
        sqp = self.normalize_sqp()

        targeting_with_share = self.join_impression_share_to_targeting(targeting, impression_share)
        joined_targeting = self.join_targeting_to_bulk(targeting, self.bulk_targets_normalized)

        account_health = self.build_account_health(targeting_with_share)
        adjusted_min_roas = account_health["adjusted_min_roas"]

        bid_recommendations = self.build_bid_recommendations(targeting_with_share, joined_targeting, adjusted_min_roas)
        search_term_actions = self.build_search_term_actions(search_terms, adjusted_min_roas)
        campaign_budget_actions = self.build_campaign_budget_actions(targeting_with_share, self.bulk_campaigns_normalized, adjusted_min_roas)
        campaign_health_dashboard = self.build_campaign_health_dashboard(targeting_with_share, adjusted_min_roas)
        sqp_opportunities, sqp_summary = self.build_sqp_opportunities(sqp, search_terms)
        smart = self.build_smart_warnings(
            targeting_with_share_df=targeting_with_share,
            search_terms_df=search_term_actions,
            campaign_health_df=campaign_health_dashboard,
            account_health=account_health,
            adjusted_min_roas=adjusted_min_roas,
            sqp_summary=sqp_summary,
        )
        preview = self.build_pre_run_preview(bid_recommendations, search_term_actions, campaign_budget_actions)

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
            "pre_run_preview": preview,
            "sqp_opportunities": sqp_opportunities,
            "sqp_summary": sqp_summary,
        }

    def process(self):
        diagnostics = self.analyze()

        search_terms = self.normalize_search_terms()
        targeting = self.normalize_targeting()
        impression_share = self.normalize_impression_share()

        targeting_with_share = self.join_impression_share_to_targeting(targeting, impression_share)
        joined_targeting = self.join_targeting_to_bulk(targeting, self.bulk_targets_normalized)

        account_health = diagnostics["account_health"]
        adjusted_min_roas = account_health["adjusted_min_roas"]

        bid_recommendations = self.build_bid_recommendations(targeting_with_share, joined_targeting, adjusted_min_roas)
        search_term_actions = self.build_search_term_actions(search_terms, adjusted_min_roas)
        campaign_budget_actions = self.build_campaign_budget_actions(targeting_with_share, self.bulk_campaigns_normalized, adjusted_min_roas)

        bid_bulk = self.generate_bid_bulk_updates(bid_recommendations)
        campaign_create_bulk = self.generate_campaign_create_bulk_updates(search_term_actions)
        ad_group_create_bulk = self.generate_ad_group_create_bulk_updates(search_term_actions)
        keyword_grad_bulk = self.generate_keyword_graduation_bulk_updates(search_term_actions)
        product_target_bulk = self.generate_product_target_bulk_updates(search_term_actions)
        negative_bulk = self.generate_negative_bulk_updates(search_term_actions)
        budget_bulk = self.generate_budget_bulk_updates(campaign_budget_actions)

        combined_bulk_updates = pd.concat(
            [bid_bulk, campaign_create_bulk, ad_group_create_bulk, keyword_grad_bulk, product_target_bulk, negative_bulk, budget_bulk],
            ignore_index=True,
        )
        combined_bulk_updates = self.apply_final_safeguards(combined_bulk_updates)

        simulation_summary = self.build_simulation_summary(combined_bulk_updates, account_health)
        self.save_run_history(simulation_summary, account_health)
        run_history = self.load_run_history()

        return {
            "combined_bulk_updates": combined_bulk_updates,
            "bid_recommendations": bid_recommendations,
            "search_term_actions": search_term_actions,
            "campaign_budget_actions": campaign_budget_actions,
            "account_health": diagnostics["account_health"],
            "account_summary": diagnostics["account_summary"],
            "campaign_health_dashboard": diagnostics["campaign_health_dashboard"],
            "smart_warnings": diagnostics["smart_warnings"],
            "optimization_suggestions": diagnostics["optimization_suggestions"],
            "pre_run_preview": diagnostics["pre_run_preview"],
            "simulation_summary": simulation_summary,
            "run_history": run_history,
            "sqp_opportunities": diagnostics["sqp_opportunities"],
            "sqp_summary": diagnostics["sqp_summary"],
        }
