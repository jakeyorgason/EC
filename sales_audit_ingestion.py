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
        high_acos_threshold=40.0,
        winning_acos_threshold=25.0,
        min_waste_spend=20.0,
        min_winner_orders=2,
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

        self.bulk_df = None
        self.impression_share_df = None
        self.targeting_df = None
        self.search_term_df = None
        self.business_report_df = None

    def build_match_type_revenue_rows(self, targeting_df):
        if targeting_df is None or targeting_df.empty:
            return []

        df = targeting_df.copy()

        if "match_type" not in df.columns:
            return []

        df["match_type"] = df["match_type"].fillna("").astype(str).str.upper().str.strip()

        base = df[df["match_type"].isin(["AUTO", "BROAD", "EXACT", "PHRASE"])].copy()

        grouped = (
            base.groupby("match_type", as_index=False)
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                spend=("spend", "sum"),
                sales=("sales", "sum"),
            )
        )

        rows = grouped.to_dict("records")

        branded_kw_sales = 0.0
        branded_kw_spend = 0.0
        branded_kw_clicks = 0.0
        branded_kw_impressions = 0.0

        return rows + [{
            "match_type": "Branded KW",
            "impressions": branded_kw_impressions,
            "clicks": branded_kw_clicks,
            "spend": branded_kw_spend,
            "sales": branded_kw_sales,
        }]


    def build_match_type_inefficient_rows(self, keyword_table):
        if keyword_table is None or keyword_table.empty:
            return []

        df = keyword_table.copy()

        if "match_type" not in df.columns:
            return []

        df["match_type"] = df["match_type"].fillna("").astype(str).str.upper().str.strip()

        inefficient = df[
            (df["spend"] >= self.min_waste_spend)
            & (
                (df["sales"] <= 0)
                | (df["acos"] > self.high_acos_threshold)
            )
        ].copy()

        grouped = (
            inefficient[inefficient["match_type"].isin(["AUTO", "BROAD", "EXACT", "PHRASE"])]
            .groupby("match_type", as_index=False)
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                spend=("spend", "sum"),
                sales=("sales", "sum"),
            )
        )

        rows = grouped.to_dict("records")

        branded_kw_sales = 0.0
        branded_kw_spend = 0.0
        branded_kw_clicks = 0.0
        branded_kw_impressions = 0.0

        return rows + [{
            "match_type": "Branded KW",
            "impressions": branded_kw_impressions,
            "clicks": branded_kw_clicks,
            "spend": branded_kw_spend,
            "sales": branded_kw_sales,
        }]

    # =========================================================
    # LOADERS
    # =========================================================
    def load_reports(self):
        self.bulk_df = self.load_bulk_sheet()
        self.impression_share_df = load_file(self.impression_share_file)
        self.targeting_df = load_file(self.targeting_file)
        self.search_term_df = load_file(self.search_term_file)
        self.business_report_df = load_file(self.business_report_file)

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

        # Use existing report ACOS/ROAS only if needed for reference;
        # final table metrics are recalculated from spend + sales.
        out = calculate_metrics(out)
        out = out[out["campaign_name"] != ""].copy()
        out = out[out["target"] != ""].copy()

        # Drop obvious duplicates just in case.
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
                df[sessions_col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
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
        # Prefer targeting as the primary account-level source.
        if targeting_df is not None and not targeting_df.empty and "sales" in targeting_df.columns:
            return float(pd.to_numeric(targeting_df["sales"], errors="coerce").fillna(0).sum())

        if search_df is not None and not search_df.empty and "sales" in search_df.columns:
            return float(pd.to_numeric(search_df["sales"], errors="coerce").fillna(0).sum())

        return 0.0

    def build_spend(self, targeting_df, search_df):
        # Prefer targeting as the primary account-level source.
        if targeting_df is not None and not targeting_df.empty and "spend" in targeting_df.columns:
            return float(pd.to_numeric(targeting_df["spend"], errors="coerce").fillna(0).sum())

        if search_df is not None and not search_df.empty and "spend" in search_df.columns:
            return float(pd.to_numeric(search_df["spend"], errors="coerce").fillna(0).sum())

        return 0.0

    def build_kpi_summary(self, targeting_df, search_df, business_df):
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
                if not reasons
                else "; ".join(reasons) + "."
            )
        )

        return {
            "status": status,
            "reasons": reasons,
            "summary": summary,
        }

    # =========================================================
    # AUDIT TABLES
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
    
    def process(self):
        self.load_reports()

        bulk_targets = self.normalize_bulk_targets()
        targeting = self.normalize_targeting()
        search_terms = self.normalize_search_terms()
        impression_share = self.normalize_impression_share()
        business = self.normalize_business_report()

        targeting_with_share = self.join_impression_share_to_targeting(targeting, impression_share)

        kpis = self.build_kpi_summary(targeting, search_terms, business)

        # Use raw targeting for spend/sales rollups
        keyword_table = self.build_keyword_spend_table(targeting)
        search_table = self.build_search_term_spend_table(search_terms)
        campaign_summary = self.build_campaign_summary(targeting)
        match_type_revenue_rows = self.build_match_type_revenue_rows(targeting)
        match_type_inefficient_rows = self.build_match_type_inefficient_rows(keyword_table)

        waste_tables = self.build_waste_tables(keyword_table, search_table)
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
            "kpi_summary": kpis,
            "campaign_summary": campaign_summary,
            "keyword_spend_table": keyword_table,
            "search_term_spend_table": search_table,
            "waste_summary": waste_summary,
            "waste_tables": waste_tables,
            "winner_tables": winner_tables,
            "health_summary": health_summary,
            "narrative": narrative,
            "match_type_revenue_rows": match_type_revenue_rows,
            "match_type_inefficient_rows": match_type_inefficient_rows,
        }

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

        # Use ONE basis only for the headline wasted spend number.
        if not st_zero_sale.empty or not st_high_acos.empty:
            wasted_spend = float(st_zero_sale["spend"].sum() + st_high_acos["spend"].sum())
            waste_basis = "search_terms"
        else:
            wasted_spend = float(kw_zero_sale["spend"].sum() + kw_high_acos["spend"].sum())
            waste_basis = "keyword_targets"

        # Safety cap: headline wasted spend can never exceed total spend.
        wasted_spend = min(wasted_spend, float(total_spend or 0))

        return {
            "keyword_zero_sale": kw_zero_sale,
            "keyword_high_acos": kw_high_acos,
            "search_zero_sale": st_zero_sale,
            "search_high_acos": st_high_acos,
            "wasted_spend": wasted_spend,
            "waste_basis": waste_basis,
        }

    def build_waste_summary(self, waste_tables, total_spend):
        wasted_spend = float(waste_tables.get("wasted_spend", 0.0))
        wasted_spend_pct = float(wasted_spend / total_spend * 100) if total_spend > 0 else 0.0

        return {
            "wasted_spend": round(wasted_spend, 2),
            "wasted_spend_pct": round(wasted_spend_pct, 2),
            "keyword_zero_sale_count": int(len(waste_tables.get("keyword_zero_sale", pd.DataFrame()))),
            "keyword_high_acos_count": int(len(waste_tables.get("keyword_high_acos", pd.DataFrame()))),
            "search_zero_sale_count": int(len(waste_tables.get("search_zero_sale", pd.DataFrame()))),
            "search_high_acos_count": int(len(waste_tables.get("search_high_acos", pd.DataFrame()))),
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

        # Keep the join available if you want ISR context later,
        # but do NOT use it as the source for spend/sales rollups.
        targeting_with_share = self.join_impression_share_to_targeting(targeting, impression_share)

        # KPI summary should use raw targeting/search/business data
        kpis = self.build_kpi_summary(targeting, search_terms, business)

        # IMPORTANT: use raw targeting here to avoid duplicated spend/sales
        keyword_table = self.build_keyword_spend_table(targeting)
        search_table = self.build_search_term_spend_table(search_terms)
        campaign_summary = self.build_campaign_summary(targeting)

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
            "kpi_summary": kpis,
            "campaign_summary": campaign_summary,
            "keyword_spend_table": keyword_table,
            "search_term_spend_table": search_table,
            "waste_summary": waste_summary,
            "waste_tables": waste_tables,
            "winner_tables": winner_tables,
            "health_summary": health_summary,
            "narrative": narrative,
        }
