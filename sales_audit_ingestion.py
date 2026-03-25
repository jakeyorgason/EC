import numpy as np
import pandas as pd

from shared_ingestion_utils import (
    calculate_metrics,
    clean_currency_series,
    clean_percent_series,
    clean_text,
    combine_preferred_columns,
    get_first_existing_column,
    get_optional_column,
    load_excel_sheet,
    load_file,
    normalize_key,
    safe_numeric,
)


class SalesAuditEngine:
    def __init__(
        self,
        bulk_file,
        impression_share_file,
        targeting_file,
        search_term_file,
        business_report_file,
        sb_campaign_file=None,
        high_acos_threshold=40.0,
        winning_acos_threshold=15.0,
        min_waste_spend=0.01,
        min_winner_orders=1,
    ):
        self.bulk_file = bulk_file
        self.impression_share_file = impression_share_file
        self.targeting_file = targeting_file
        self.search_term_file = search_term_file
        self.business_report_file = business_report_file

        self.high_acos_threshold = float(high_acos_threshold) / 100.0
        self.winning_acos_threshold = float(winning_acos_threshold) / 100.0
        self.min_waste_spend = float(min_waste_spend)
        self.min_winner_orders = int(min_winner_orders)
        self.sb_campaign_file = sb_campaign_file
        self.sb_campaign_df = None

        self.bulk_df = None
        self.impression_share_df = None
        self.targeting_df = None
        self.search_term_df = None
        self.business_report_df = None

    # =========================================================
    # LOADERS
    # =========================================================
    def load_reports(self):
        self.bulk_df = self.load_bulk_sheet()
        self.impression_share_df = load_file(self.impression_share_file)
        self.targeting_df = load_file(self.targeting_file)
        self.search_term_df = load_file(self.search_term_file)
        self.business_report_df = load_file(self.business_report_file)
        self.sb_campaign_df = load_file(self.sb_campaign_file) if self.sb_campaign_file is not None else pd.DataFrame()

    def load_bulk_sheet(self):
        try:
            return load_excel_sheet(
                self.bulk_file,
                sheet_name="Sponsored Products Campaigns",
                dtype=str,
            )
        except Exception:
            return load_file(self.bulk_file)

    # =========================================================
    # NORMALIZATION
    # =========================================================
    def normalize_bulk_targets(self):
        if self.bulk_df is None or self.bulk_df.empty:
            return pd.DataFrame()

        df = self.bulk_df.copy()

        entity_col = get_optional_column(df, ["Entity"])
        if entity_col is not None:
            df = df[df[entity_col].astype(str).isin(["Keyword", "Product Targeting"])].copy()

        out = pd.DataFrame(index=df.index)

        out["entity"] = clean_text(df[get_optional_column(df, ["Entity"])]) if get_optional_column(df, ["Entity"]) else ""
        out["campaign_id"] = clean_text(df[get_optional_column(df, ["Campaign ID"])]) if get_optional_column(df, ["Campaign ID"]) else ""
        out["ad_group_id"] = clean_text(df[get_optional_column(df, ["Ad Group ID"])]) if get_optional_column(df, ["Ad Group ID"]) else ""
        out["keyword_id"] = clean_text(df[get_optional_column(df, ["Keyword ID"])]) if get_optional_column(df, ["Keyword ID"]) else ""
        out["campaign_name"] = clean_text(df[get_first_existing_column(df, ["Campaign Name"], "campaign name")])
        out["ad_group_name"] = clean_text(df[get_optional_column(df, ["Ad Group Name"])]) if get_optional_column(df, ["Ad Group Name"]) else ""
        out["target"] = combine_preferred_columns(
            df,
            primary_candidates=["Keyword Text", "Targeting"],
            fallback_candidates=["Product Targeting Expression", "Resolved Expression"],
            label="bulk target",
        )
        out["match_type"] = clean_text(df[get_optional_column(df, ["Match Type"])]) if get_optional_column(df, ["Match Type"]) else ""
        out["current_bid"] = safe_numeric(df[get_optional_column(df, ["Bid"])]) if get_optional_column(df, ["Bid"]) else 0.0
        out["state"] = clean_text(df[get_optional_column(df, ["State"])]) if get_optional_column(df, ["State"]) else ""

        out = out[out["campaign_name"] != ""].copy()
        out = out[out["target"] != ""].copy()
        out = out.drop_duplicates().reset_index(drop=True)
        return out

    def normalize_targeting(self):
        if self.targeting_df is None or self.targeting_df.empty:
            return pd.DataFrame()

        df = self.targeting_df.copy()
        out = pd.DataFrame(index=df.index)

        out["campaign_name"] = clean_text(df[get_first_existing_column(df, ["Campaign Name"], "campaign name")])
        out["ad_group_name"] = clean_text(df[get_first_existing_column(df, ["Ad Group Name"], "ad group name")])
        out["target"] = clean_text(
            df[get_first_existing_column(
                df,
                ["Targeting", "Keyword Text", "Target", "Product Targeting Expression", "Resolved Expression"],
                "target",
            )]
        )

        match_col = get_optional_column(df, ["Match Type"])
        out["match_type"] = clean_text(df[match_col]) if match_col else ""

        out["impressions"] = safe_numeric(df[get_first_existing_column(df, ["Impressions"], "impressions")])
        out["clicks"] = safe_numeric(df[get_first_existing_column(df, ["Clicks"], "clicks")])

        spend_col = get_first_existing_column(df, ["Spend"], "spend")
        sales_col = get_first_existing_column(df, ["7 Day Total Sales ", "7 Day Total Sales", "Sales"], "sales")
        orders_col = get_first_existing_column(df, ["7 Day Total Orders (#)", "Orders", "7 Day Total Orders"], "orders")

        out["spend"] = self._parse_money_or_numeric(df[spend_col])
        out["sales"] = self._parse_money_or_numeric(df[sales_col])
        out["orders"] = safe_numeric(df[orders_col])

        out = calculate_metrics(out)
        out = out[out["campaign_name"] != ""].copy()
        out = out[out["target"] != ""].copy()

        out = out.drop_duplicates(
            subset=["campaign_name", "ad_group_name", "target", "match_type", "impressions", "clicks", "spend", "sales", "orders"]
        ).reset_index(drop=True)

        return out

    def normalize_search_terms(self):
        if self.search_term_df is None or self.search_term_df.empty:
            return pd.DataFrame()

        df = self.search_term_df.copy()
        out = pd.DataFrame(index=df.index)

        out["campaign_name"] = clean_text(df[get_first_existing_column(df, ["Campaign Name"], "campaign name")])
        out["ad_group_name"] = clean_text(df[get_first_existing_column(df, ["Ad Group Name"], "ad group name")])
        out["customer_search_term"] = clean_text(
            df[get_first_existing_column(df, ["Customer Search Term", "Search Term"], "customer search term")]
        )

        match_col = get_optional_column(df, ["Match Type"])
        out["match_type"] = clean_text(df[match_col]) if match_col else ""

        out["impressions"] = safe_numeric(df[get_first_existing_column(df, ["Impressions"], "impressions")])
        out["clicks"] = safe_numeric(df[get_first_existing_column(df, ["Clicks"], "clicks")])

        spend_col = get_first_existing_column(df, ["Spend"], "spend")
        sales_col = get_first_existing_column(df, ["7 Day Total Sales ", "7 Day Total Sales", "Sales"], "sales")
        orders_col = get_first_existing_column(df, ["7 Day Total Orders (#)", "Orders", "7 Day Total Orders"], "orders")

        out["spend"] = self._parse_money_or_numeric(df[spend_col])
        out["sales"] = self._parse_money_or_numeric(df[sales_col])
        out["orders"] = safe_numeric(df[orders_col])

        out = calculate_metrics(out)
        out = out[out["campaign_name"] != ""].copy()
        out = out[out["customer_search_term"] != ""].copy()

        out = out.drop_duplicates(
            subset=["campaign_name", "ad_group_name", "customer_search_term", "match_type", "impressions", "clicks", "spend", "sales", "orders"]
        ).reset_index(drop=True)

        return out

    def normalize_impression_share(self):
        if self.impression_share_df is None or self.impression_share_df.empty:
            return pd.DataFrame()

        df = self.impression_share_df.copy()
        out = pd.DataFrame(index=df.index)

        out["campaign_name"] = clean_text(df[get_first_existing_column(df, ["Campaign Name"], "campaign name")])

        target_col = get_optional_column(
            df,
            ["Targeting", "Keyword Text", "Target", "Customer Search Term"],
        )
        out["target"] = clean_text(df[target_col]) if target_col else ""

        match_col = get_optional_column(df, ["Match Type"])
        out["match_type"] = clean_text(df[match_col]) if match_col else ""

        share_col = get_first_existing_column(
            df,
            [
                "Top-of-search Impression Share",
                "Impression Share",
                "Search top impression share",
                "Search Term Impression Share",
            ],
            "impression share",
        )
        out["impression_share_pct"] = clean_percent_series(df[share_col])

        out = out[out["campaign_name"] != ""].copy()
        out = out.drop_duplicates().reset_index(drop=True)
        return out

    def normalize_business_report(self):
        if self.business_report_df is None or self.business_report_df.empty:
            return pd.DataFrame()

        df = self.business_report_df.copy()

        sales_col = get_optional_column(
            df,
            ["Ordered Product Sales", "Sales", "Ordered Sales", "Total Sales"],
        )
        sessions_col = get_optional_column(
            df,
            ["Sessions - Total", "Sessions", "Total Sessions", "Browser Sessions", "Sessions Total"],
        )
        units_col = get_optional_column(
            df,
            ["Units Ordered", "Ordered Product Sales Units", "Units"],
        )
        date_col = get_optional_column(df, ["Date", "Day", "Report Date"])

        if sales_col is None:
            return pd.DataFrame()

        out = pd.DataFrame(index=df.index)
        out["total_sales"] = self._parse_money_or_numeric(df[sales_col])

        if sessions_col:
            sessions_series = (
                df[sessions_col].astype(str).str.replace(",", "", regex=False).str.strip()
            )
            out["sessions"] = pd.to_numeric(sessions_series, errors="coerce").fillna(0)
        else:
            out["sessions"] = 0.0

        if units_col:
            out["units_ordered"] = safe_numeric(df[units_col])
        else:
            out["units_ordered"] = 0.0

        out["date"] = clean_text(df[date_col]) if date_col else ""

        return out.reset_index(drop=True)

    def normalize_sb_campaign_report(self):
        if self.sb_campaign_df is None or self.sb_campaign_df.empty:
            return pd.DataFrame()
    
        df = self.sb_campaign_df.copy()
        out = pd.DataFrame(index=df.index)
    
        def first_present(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None
    
        sales_col = first_present([
            "Sales",
            "14 Day Total Sales",
            "14 day total sales",
            "Attributed Sales",
        ])
        orders_col = first_present([
            "Orders",
            "14 Day Total Orders",
            "14 day total orders",
        ])
        ntb_sales_col = first_present([
            "New-to-brand sales",
            "14 Day New-to-brand Sales",
            "14 day new-to-brand sales",
            "NTB Sales",
        ])
        ntb_orders_col = first_present([
            "New-to-brand orders",
            "14 Day New-to-brand Orders",
            "14 day new-to-brand orders",
            "NTB Orders",
        ])
    
        out["sales"] = self._parse_money_or_numeric(df[sales_col]) if sales_col else 0.0
        out["orders"] = safe_numeric(df[orders_col]) if orders_col else 0.0
        out["ntb_sales"] = self._parse_money_or_numeric(df[ntb_sales_col]) if ntb_sales_col else 0.0
        out["ntb_orders"] = safe_numeric(df[ntb_orders_col]) if ntb_orders_col else 0.0
    
        return out.reset_index(drop=True)

    # =========================================================
    # HELPERS
    # =========================================================
    def _parse_money_or_numeric(self, series):
        text_series = series.astype(str)
        if text_series.str.contains(r"[$,]").any():
            return clean_currency_series(series)
        return safe_numeric(series)

    # =========================================================
    # JOINS
    # =========================================================
    def join_impression_share_to_targeting(self, targeting_df, share_df):
        if targeting_df is None or targeting_df.empty:
            return pd.DataFrame()

        out = targeting_df.copy()
        out["impression_share_pct"] = 0.0

        if share_df is None or share_df.empty:
            return out

        left = out.copy()
        right = share_df.copy()

        left["_campaign_key"] = left["campaign_name"].map(normalize_key)
        left["_target_key"] = left["target"].map(normalize_key)
        left["_match_key"] = left["match_type"].map(normalize_key)

        right["_campaign_key"] = right["campaign_name"].map(normalize_key)
        right["_target_key"] = right["target"].map(normalize_key)
        right["_match_key"] = right["match_type"].map(normalize_key)

        merged = left.merge(
            right[["_campaign_key", "_target_key", "_match_key", "impression_share_pct"]].drop_duplicates(),
            on=["_campaign_key", "_target_key", "_match_key"],
            how="left",
            suffixes=("", "_isr"),
        )

        merged["impression_share_pct"] = merged["impression_share_pct_isr"].fillna(0.0)
        merged = merged.drop(columns=["_campaign_key", "_target_key", "_match_key", "impression_share_pct_isr"])

        return merged

    # =========================================================
    # KPI BUILDERS
    # =========================================================
    def build_total_sales(self, business_df):
        if business_df is None or business_df.empty or "total_sales" not in business_df.columns:
            return 0.0
        return float(pd.to_numeric(business_df["total_sales"], errors="coerce").fillna(0).sum())

    def build_ad_sales(self, targeting_df, search_df):
        if targeting_df is not None and not targeting_df.empty and "sales" in targeting_df.columns:
            return float(pd.to_numeric(targeting_df["sales"], errors="coerce").fillna(0).sum())
        if search_df is not None and not search_df.empty and "sales" in search_df.columns:
            return float(pd.to_numeric(search_df["sales"], errors="coerce").fillna(0).sum())
        return 0.0

    def build_spend(self, targeting_df, search_df):
        if targeting_df is not None and not targeting_df.empty and "spend" in targeting_df.columns:
            return float(pd.to_numeric(targeting_df["spend"], errors="coerce").fillna(0).sum())
        if search_df is not None and not search_df.empty and "spend" in search_df.columns:
            return float(pd.to_numeric(search_df["spend"], errors="coerce").fillna(0).sum())
        return 0.0

    def build_kpi_summary(self, targeting_df, search_df, business_df, sb_campaign_df):
        spend = self.build_spend(targeting_df, search_df)
        ad_sales = self.build_ad_sales(targeting_df, search_df)
        total_sales = self.build_total_sales(business_df)
        organic_sales = total_sales - ad_sales

        acos = float(spend / ad_sales) if ad_sales > 0 else 0.0
        roas = float(ad_sales / spend) if spend > 0 else 0.0
        tacos = float(spend / total_sales) if total_sales > 0 else 0.0
        organic_share = float(organic_sales / total_sales) if total_sales > 0 else 0.0

        sessions = 0.0
        if business_df is not None and not business_df.empty and "sessions" in business_df.columns:
            sessions = float(pd.to_numeric(business_df["sessions"], errors="coerce").fillna(0).sum())

        units_ordered = 0.0
        if business_df is not None and not business_df.empty and "units_ordered" in business_df.columns:
            units_ordered = float(pd.to_numeric(business_df["units_ordered"], errors="coerce").fillna(0).sum())

        unit_session_percentage = float(units_ordered / sessions) if sessions > 0 else 0.0

        ntb_sales = 0.0
        ntb_orders = 0.0
        
        if sb_campaign_df is not None and not sb_campaign_df.empty:
            if "ntb_sales" in sb_campaign_df.columns:
                ntb_sales = float(pd.to_numeric(sb_campaign_df["ntb_sales"], errors="coerce").fillna(0).sum())
            if "ntb_orders" in sb_campaign_df.columns:
                ntb_orders = float(pd.to_numeric(sb_campaign_df["ntb_orders"], errors="coerce").fillna(0).sum())

        ntb_sales_pct = float(ntb_sales / ad_sales * 100) if ad_sales > 0 else 0.0
        ntb_orders_pct = float(ntb_orders / units_ordered * 100) if units_ordered > 0 else 0.0

        return {
            "spend": round(spend, 2),
            "ad_sales": round(ad_sales, 2),
            "total_sales": round(total_sales, 2),
            "organic_sales": round(organic_sales, 2),
            "acos_pct": round(acos * 100, 2),
            "roas": round(roas, 2),
            "tacos_pct": round(tacos * 100, 2),
            "organic_share_pct": round(organic_share * 100, 2),
            "sessions": round(sessions, 2),
            "units_ordered": round(units_ordered, 2),
            "estimated_post_ad_contribution": round(ad_sales - spend, 2),
            "unit_session_percentage": round(unit_session_percentage * 100, 2),
            "ntb_sales": round(ntb_sales, 2),
            "ntb_orders": round(ntb_orders, 2),
            "ntb_sales_pct": round(ntb_sales_pct, 2),
            "ntb_orders_pct": round(ntb_orders_pct, 2),
        }

    def build_account_health_summary(self, kpis, waste_summary):
        acos = kpis.get("acos_pct", 0)
        tacos = kpis.get("tacos_pct", 0)
        organic_share = kpis.get("organic_share_pct", 0)
        waste_pct = waste_summary.get("wasted_spend_pct", 0)

        status = "Healthy"
        reasons = []

        if acos > 40:
            reasons.append("ACOS is above 40%")
        if tacos > 20:
            reasons.append("TACOS is above 20%")
        if organic_share < 40:
            reasons.append("organic share is below 40%")
        if waste_pct > 30:
            reasons.append("wasted spend is above 30%")

        if reasons:
            status = "At Risk"
        else:
            mixed_reasons = []
            if acos > 30:
                mixed_reasons.append("ACOS is elevated")
            if tacos > 15:
                mixed_reasons.append("TACOS is elevated")
            if organic_share < 50:
                mixed_reasons.append("organic share is modest")
            if waste_pct > 20:
                mixed_reasons.append("wasted spend is elevated")
            if mixed_reasons:
                status = "Mixed"
                reasons = mixed_reasons

        summary = (
            f"{status}: "
            + (
                "The account shows generally acceptable efficiency and organic support."
                if not reasons else "; ".join(reasons) + "."
            )
        )

        return {
            "status": status,
            "reasons": reasons,
            "summary": summary,
        }

    # =========================================================
    # TABLE BUILDERS
    # =========================================================
    def build_keyword_spend_table(self, targeting_df):
        if targeting_df is None or targeting_df.empty:
            return pd.DataFrame()

        agg_map = {
            "impressions": ("impressions", "sum"),
            "clicks": ("clicks", "sum"),
            "spend": ("spend", "sum"),
            "sales": ("sales", "sum"),
            "orders": ("orders", "sum"),
        }

        if "impression_share_pct" in targeting_df.columns:
            agg_map["impression_share_pct"] = ("impression_share_pct", "mean")

        grouped = (
            targeting_df.groupby(
                ["campaign_name", "ad_group_name", "target", "match_type"],
                as_index=False,
            )
            .agg(**agg_map)
        )

        grouped = calculate_metrics(grouped)
        grouped["acos_pct"] = grouped["acos"] * 100
        grouped = grouped.sort_values(["spend", "sales"], ascending=[False, False]).reset_index(drop=True)
        return grouped

    def build_search_term_spend_table(self, search_df):
        if search_df is None or search_df.empty:
            return pd.DataFrame()

        grouped = (
            search_df.groupby(
                ["campaign_name", "ad_group_name", "customer_search_term", "match_type"],
                as_index=False,
            )
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                spend=("spend", "sum"),
                sales=("sales", "sum"),
                orders=("orders", "sum"),
            )
        )

        grouped = calculate_metrics(grouped)
        grouped["acos_pct"] = grouped["acos"] * 100
        grouped = grouped.sort_values(["spend", "sales"], ascending=[False, False]).reset_index(drop=True)
        return grouped

    def build_waste_tables(self, keyword_table, search_table, total_spend):
        kw_zero_sale = pd.DataFrame()
        kw_high_acos = pd.DataFrame()
        st_zero_sale = pd.DataFrame()
        st_high_acos = pd.DataFrame()
    
        if keyword_table is not None and not keyword_table.empty:
            kw_zero_sale = keyword_table[
                (keyword_table["spend"] >= self.min_waste_spend) & (keyword_table["sales"] <= 0)
            ].copy()
    
            kw_high_acos = keyword_table[
                (keyword_table["spend"] >= self.min_waste_spend)
                & (keyword_table["sales"] > 0)
                & (keyword_table["acos"] > self.high_acos_threshold)
            ].copy()
    
        if search_table is not None and not search_table.empty:
            st_zero_sale = search_table[
                (search_table["spend"] >= self.min_waste_spend) & (search_table["sales"] <= 0)
            ].copy()
    
            st_high_acos = search_table[
                (search_table["spend"] >= self.min_waste_spend)
                & (search_table["sales"] > 0)
                & (search_table["acos"] > self.high_acos_threshold)
            ].copy()
    
        kw_zero_sale = kw_zero_sale.sort_values("spend", ascending=False).reset_index(drop=True)
        kw_high_acos = kw_high_acos.sort_values("spend", ascending=False).reset_index(drop=True)
        st_zero_sale = st_zero_sale.sort_values("spend", ascending=False).reset_index(drop=True)
        st_high_acos = st_high_acos.sort_values("spend", ascending=False).reset_index(drop=True)
    
        def calculate_wasted_spend(zero_df, high_df):
            zero_waste = float(zero_df["spend"].sum()) if not zero_df.empty else 0.0
    
            high_waste = 0.0
            if not high_df.empty:
                allowed_spend = high_df["sales"] * self.high_acos_threshold
                wasted_portion = (high_df["spend"] - allowed_spend).clip(lower=0)
                high_waste = float(wasted_portion.sum())
    
            return zero_waste + high_waste
    
        if not st_zero_sale.empty or not st_high_acos.empty:
            wasted_spend = calculate_wasted_spend(st_zero_sale, st_high_acos)
        else:
            wasted_spend = calculate_wasted_spend(kw_zero_sale, kw_high_acos)
    
        wasted_spend = min(wasted_spend, float(total_spend or 0))
    
        return {
            "keyword_zero_sale": kw_zero_sale,
            "keyword_high_acos": kw_high_acos,
            "search_zero_sale": st_zero_sale,
            "search_high_acos": st_high_acos,
            "wasted_spend": wasted_spend,
        }

    def build_waste_summary(self, waste_tables, total_spend):
        wasted_spend = float(waste_tables.get("wasted_spend", 0.0))
        wasted_spend_pct = float(wasted_spend / total_spend * 100) if total_spend > 0 else 0.0

        spend_no_sale = 0.0
        st_zero = waste_tables.get("search_zero_sale", pd.DataFrame())
        kw_zero = waste_tables.get("keyword_zero_sale", pd.DataFrame())

        if isinstance(st_zero, pd.DataFrame) and not st_zero.empty:
            spend_no_sale = float(st_zero["spend"].sum())
        elif isinstance(kw_zero, pd.DataFrame) and not kw_zero.empty:
            spend_no_sale = float(kw_zero["spend"].sum())

        return {
            "wasted_spend": round(wasted_spend, 2),
            "wasted_spend_pct": round(wasted_spend_pct, 2),
            "spend_no_sale": round(spend_no_sale, 2),
        }

    def build_winner_tables(self, keyword_table, search_table):
        kw_winners = pd.DataFrame()
        st_winners = pd.DataFrame()

        if keyword_table is not None and not keyword_table.empty:
            kw_winners = keyword_table[
                (keyword_table["orders"] >= self.min_winner_orders)
                & (keyword_table["sales"] > 0)
                & (keyword_table["acos"] <= self.winning_acos_threshold)
            ].copy().sort_values(["acos", "sales"], ascending=[True, False])

        if search_table is not None and not search_table.empty:
            st_winners = search_table[
                (search_table["orders"] >= self.min_winner_orders)
                & (search_table["sales"] > 0)
                & (search_table["acos"] <= self.winning_acos_threshold)
            ].copy().sort_values(["acos", "sales"], ascending=[True, False])

        return {
            "keyword_winners": kw_winners.reset_index(drop=True),
            "search_winners": st_winners.reset_index(drop=True),
        }

    def build_campaign_summary(self, targeting_df):
        if targeting_df is None or targeting_df.empty:
            return pd.DataFrame()

        agg_map = {
            "spend": ("spend", "sum"),
            "sales": ("sales", "sum"),
            "orders": ("orders", "sum"),
        }

        if "impression_share_pct" in targeting_df.columns:
            agg_map["impression_share_pct"] = ("impression_share_pct", "mean")

        grouped = (
            targeting_df.groupby("campaign_name", as_index=False)
            .agg(**agg_map)
        )

        grouped = calculate_metrics(grouped)
        grouped["acos_pct"] = grouped["acos"] * 100

        grouped["campaign_status"] = np.select(
            [
                (grouped["spend"] >= 100) & (grouped["orders"] == 0),
                (grouped["acos"] > self.high_acos_threshold) & (grouped["sales"] > 0),
                (grouped["orders"] >= self.min_winner_orders) & (grouped["acos"] <= self.winning_acos_threshold),
            ],
            ["Waste Alert", "Inefficient", "Winning"],
            default="Stable",
        )

        return grouped.sort_values(["spend", "sales"], ascending=[False, False]).reset_index(drop=True)

    def build_date_range_label(self, targeting_df, search_df, business_df):
        date_values = []

        for df in [targeting_df, search_df, business_df]:
            if df is None or df.empty:
                continue

            for col in ["Start Date", "End Date", "date", "Date", "Day", "Report Date"]:
                if col in df.columns:
                    parsed = pd.to_datetime(df[col], errors="coerce").dropna()
                    if not parsed.empty:
                        date_values.extend(parsed.tolist())

        if not date_values:
            return ""

        start_date = min(date_values)
        end_date = max(date_values)
        return f"{start_date:%m/%d} - {end_date:%m/%d}"

    def is_branded_term(self, term: str, brand_name: str) -> bool:
        term_text = str(term or "").strip().lower()
        brand_text = str(brand_name or "").strip().lower()

        if not term_text or not brand_text:
            return False

        return brand_text in term_text
    
    def build_match_type_revenue_rows(self, search_df, brand_name: str):
        if search_df is None or search_df.empty:
            return []

        df = search_df.copy()

        if "match_type" not in df.columns:
            df["match_type"] = ""

        if "customer_search_term" not in df.columns:
            df["customer_search_term"] = ""

        df["match_type"] = (
            df["match_type"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        df["customer_search_term"] = (
            df["customer_search_term"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    
        def bucket_row(row):
            term = row["customer_search_term"]
            match_type = row["match_type"]
    
            if self.is_branded_term(term, brand_name):
                return "Branded KW"
            if match_type == "EXACT":
                return "EXACT"
            if match_type == "PHRASE":
                return "PHRASE"
            return "AUTO"
    
        df["match_bucket"] = df.apply(bucket_row, axis=1)
    
        grouped = (
            df.groupby("match_bucket", as_index=False)
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                spend=("spend", "sum"),
                sales=("sales", "sum"),
            )
        )
    
        row_order = ["AUTO", "BROAD", "EXACT", "PHRASE", "Branded KW"]
        row_map = {row["match_bucket"]: row for _, row in grouped.iterrows()}
    
        rows = []
        for label in row_order:
            row = row_map.get(label)
            if row is None:
                rows.append({
                    "match_type": label,
                    "impressions": 0,
                    "clicks": 0,
                    "spend": 0,
                    "sales": 0,
                })
            else:
                rows.append({
                    "match_type": label,
                    "impressions": float(row["impressions"]),
                    "clicks": float(row["clicks"]),
                    "spend": float(row["spend"]),
                    "sales": float(row["sales"]),
                })
    
        return rows

    def build_match_type_inefficient_rows(self, search_df, brand_name: str):
        if search_df is None or search_df.empty:
            return []
    
        df = search_df.copy()
    
        if "match_type" not in df.columns:
            df["match_type"] = ""
    
        if "customer_search_term" not in df.columns:
            df["customer_search_term"] = ""
    
        df["match_type"] = (
            df["match_type"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )
    
        df["customer_search_term"] = (
            df["customer_search_term"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    
        inefficient = df[
            (df["spend"] >= self.min_waste_spend)
            & ((df["sales"] <= 0) | (df["acos"] > self.high_acos_threshold))
        ].copy()
    
        def bucket_row(row):
            term = row["customer_search_term"]
            match_type = row["match_type"]
    
            if self.is_branded_term(term, brand_name):
                return "Branded KW"
            if match_type == "EXACT":
                return "EXACT"
            if match_type == "PHRASE":
                return "PHRASE"
            return "AUTO"
    
        inefficient["match_bucket"] = inefficient.apply(bucket_row, axis=1)
    
        grouped = (
            inefficient.groupby("match_bucket", as_index=False)
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                spend=("spend", "sum"),
                sales=("sales", "sum"),
            )
        )
    
        row_order = ["AUTO", "BROAD", "EXACT", "PHRASE", "Branded KW"]
        row_map = {row["match_bucket"]: row for _, row in grouped.iterrows()}
    
        rows = []
        for label in row_order:
            row = row_map.get(label)
            if row is None:
                rows.append({
                    "match_type": label,
                    "impressions": 0,
                    "clicks": 0,
                    "spend": 0,
                    "sales": 0,
                })
            else:
                rows.append({
                    "match_type": label,
                    "impressions": float(row["impressions"]),
                    "clicks": float(row["clicks"]),
                    "spend": float(row["spend"]),
                    "sales": float(row["sales"]),
                })
    
        return rows

    def build_campaign_type_rows(self, targeting_df):
        if targeting_df is None or targeting_df.empty or "campaign_name" not in targeting_df.columns:
            return []

        df = targeting_df.copy()

        def infer_campaign_type(name: str) -> str:
            text = str(name).upper()
            if "SPONSORED PRODUCTS" in text or "| SP " in text or text.startswith("SP ") or " SP " in text:
                return "SP"
            if "SPONSORED BRANDS" in text or "| SB " in text or text.startswith("SB ") or " SB " in text:
                return "SB"
            if "SPONSORED DISPLAY" in text or "| SD " in text or text.startswith("SD ") or " SD " in text:
                return "SD"
            return "SP"

        df["campaign_type"] = df["campaign_name"].map(infer_campaign_type)

        grouped = (
            df.groupby("campaign_type", as_index=False)
            .agg(sales=("sales", "sum"))
        )

        return grouped.to_dict("records")

    def build_narrative(self, kpis, waste_summary, health_summary, winner_tables):
        kw_winners = len(winner_tables.get("keyword_winners", pd.DataFrame()))
        st_winners = len(winner_tables.get("search_winners", pd.DataFrame()))

        return (
            f"The account generated ${kpis['ad_sales']:,.2f} in ad sales on ${kpis['spend']:,.2f} "
            f"of spend, producing {kpis['roas']:.2f} ROAS and {kpis['acos_pct']:.2f}% ACOS. "
            f"Total sales were ${kpis['total_sales']:,.2f}, giving {kpis['tacos_pct']:.2f}% TACOS "
            f"and ${kpis['organic_sales']:,.2f} in estimated organic sales. "
            f"Wasted spend is estimated at ${waste_summary['wasted_spend']:,.2f} "
            f"({waste_summary['wasted_spend_pct']:.2f}% of spend). "
            f"Account verdict: {health_summary['status']}. "
            f"Winning terms identified: {kw_winners} keyword targets and {st_winners} customer search terms."
        )

    # =========================================================
    # PROCESS
    # =========================================================
    def process(self):
        self.load_reports()

        bulk_targets = self.normalize_bulk_targets()
        targeting = self.normalize_targeting()
        search_terms = self.normalize_search_terms()
        impression_share = self.normalize_impression_share()
        business = self.normalize_business_report()
        sb_campaign = self.normalize_sb_campaign_report()

        targeting_with_share = self.join_impression_share_to_targeting(targeting, impression_share)

        kpis = self.build_kpi_summary(targeting, search_terms, business, sb_campaign)
        keyword_table = self.build_keyword_spend_table(targeting)
        search_table = self.build_search_term_spend_table(search_terms)
        campaign_summary = self.build_campaign_summary(targeting)
        date_range_label = self.build_date_range_label(targeting, search_terms, business)

        brand_name = getattr(self, "brand_name", "")

        match_type_revenue_rows = self.build_match_type_revenue_rows(search_terms, brand_name)
        match_type_inefficient_rows = self.build_match_type_inefficient_rows(search_terms, brand_name)
        campaign_type_rows = self.build_campaign_type_rows(targeting)

        waste_tables = self.build_waste_tables(keyword_table, search_table, total_spend=kpis["spend"])
        waste_summary = self.build_waste_summary(waste_tables, total_spend=kpis["spend"])

        winner_tables = self.build_winner_tables(keyword_table, search_table)
        health_summary = self.build_account_health_summary(kpis, waste_summary)
        narrative = self.build_narrative(kpis, waste_summary, health_summary, winner_tables)

        return {
            "bulk_targets": bulk_targets,
            "targeting": targeting,
            "targeting_with_share": targeting_with_share,
            "search_terms": search_terms,
            "impression_share": impression_share,
            "business_report": business,
            "sb_campaign_report": sb_campaign,
            "kpi_summary": kpis,
            "campaign_summary": campaign_summary,
            "campaign_type_rows": campaign_type_rows,
            "keyword_spend_table": keyword_table,
            "search_term_spend_table": search_table,
            "waste_summary": waste_summary,
            "waste_tables": waste_tables,
            "winner_tables": winner_tables,
            "health_summary": health_summary,
            "narrative": narrative,
            "date_range_label": date_range_label,
            "match_type_revenue_rows": match_type_revenue_rows,
            "match_type_inefficient_rows": match_type_inefficient_rows,
        }
