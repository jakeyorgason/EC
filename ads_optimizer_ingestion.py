
import io
import os
import calendar
from datetime import date

import numpy as np
import pandas as pd

def find_matching_sheet_name(sheet_names, candidates):
    normalized = {str(name).strip().lower(): name for name in sheet_names}

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]

    for original_name in sheet_names:
        name = str(original_name).strip().lower()
        for candidate in candidates:
            candidate_key = str(candidate).strip().lower()
            if candidate_key in name or name in candidate_key:
                return original_name

    return None

SP_BULK_SHEET_CANDIDATES = [
    "Sponsored Products Campaigns",
    "Sponsored Products",
    "SP Campaigns",
    "SP",
]

SB_BULK_SHEET_CANDIDATES = [
    "Sponsored Brands Campaigns",
    "Sponsored Brands",
    "SB Campaigns",
    "SB",
    "SB Multi Ad Group Campaigns",
]

SD_BULK_SHEET_CANDIDATES = [
    "Sponsored Display Campaigns",
    "Sponsored Display",
    "SD Campaigns",
    "SD",
    "RAS Campaigns",
]

def safe_concat_frames(frames, ignore_index=True):
    """Concatenate only non-empty DataFrames and never raise on an empty input list."""
    valid_frames = []
    if frames is None:
        return pd.DataFrame()
    for frame in frames:
        if frame is None:
            continue
        if isinstance(frame, pd.DataFrame):
            if not frame.empty:
                valid_frames.append(frame)
            continue
        try:
            candidate = pd.DataFrame(frame)
            if not candidate.empty:
                valid_frames.append(candidate)
        except Exception:
            continue
    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=ignore_index)



def ensure_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a lowercase numeric score column so sorting never fails."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    out = df.copy()
    if "score" not in out.columns and "Score" in out.columns:
        out["score"] = out["Score"]
    if "score" not in out.columns:
        out["score"] = 0.0

    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    return out


def ensure_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a lowercase numeric score column so diagnostics and sorting never fail."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    out = df.copy()
    if "score" not in out.columns and "Score" in out.columns:
        out["score"] = out["Score"]
    if "score" not in out.columns:
        out["score"] = 0.0

    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    return out


def ensure_trend_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee trend/memory columns exist so diagnostics and processing never fail."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    out = df.copy()

    defaults = {
        "recent_action_count": 0,
        "prior_action_direction": "",
        "cooldown_active": False,
        "last_action_days_ago": 999,
        "previous_roas": 0.0,
        "previous_clicks": 0.0,
        "previous_orders": 0.0,
        "previous_score": 0.0,
        "roas_trend": "flat",
        "click_trend": "flat",
        "order_trend": "flat",
    }

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out["recent_action_count"] = pd.to_numeric(out["recent_action_count"], errors="coerce").fillna(0).astype(int)
    out["last_action_days_ago"] = pd.to_numeric(out["last_action_days_ago"], errors="coerce").fillna(999)
    for col in ["previous_roas", "previous_clicks", "previous_orders", "previous_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["cooldown_active"] = out["cooldown_active"].fillna(False).astype(bool)

    return out

def _dedupe_and_strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and keep the first occurrence of any duplicate column names."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")].copy()
    return out

def apply_cross_type_bulk_safeguards(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = _dedupe_and_strip_columns(df)

    preferred_columns = [
        'Product', 'Entity', 'Operation', 'Campaign ID', 'Ad Group ID', 'Keyword ID',
        'Product Targeting ID', 'Campaign Name', 'Ad Group Name', 'State',
        'Keyword Text', 'Match Type', 'Bid', 'Budget', 'Daily Budget', 'Placement Type', 'Placement %', 'Optimizer Action',
        'ad_type', 'source_type', 'campaign', 'ad_group'
    ]

    for col in preferred_columns:
        if col not in out.columns:
            out[col] = ''

    ordered_cols = preferred_columns + [c for c in out.columns if c not in preferred_columns]
    ordered_cols = list(dict.fromkeys(ordered_cols))
    out = out[ordered_cols]

    text_cols = [
        'Product', 'Entity', 'Operation', 'Campaign ID', 'Ad Group ID', 'Keyword ID',
        'Product Targeting ID', 'Campaign Name', 'Ad Group Name', 'State',
        'Keyword Text', 'Match Type', 'Optimizer Action', 'ad_type', 'source_type',
        'campaign', 'ad_group'
    ]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].fillna('').astype(str).str.replace(r'\.0$', '', regex=True)

    for col in ['Keyword Text', 'Match Type', 'Campaign Name', 'Ad Group Name']:
        if col in out.columns:
            out[col] = out[col].fillna('').astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

    signature_cols = [
        'Product', 'Entity', 'Operation', 'Campaign ID', 'Ad Group ID', 'Keyword ID',
        'Product Targeting ID', 'Campaign Name', 'Ad Group Name', 'Keyword Text', 'Match Type',
        'Bid', 'Budget', 'Daily Budget'
    ]
    existing_signature_cols = [c for c in signature_cols if c in out.columns]
    if existing_signature_cols:
        out = out.drop_duplicates(subset=existing_signature_cols, keep='last')
    else:
        out = out.drop_duplicates()

    out = _dedupe_and_strip_columns(out)
    return out.reset_index(drop=True)


# =========================================================
# Advanced optimization helpers
# =========================================================
AMAZON_ONLY_MODE = True

BRAND_CLASSIFICATION_HINTS = [
    "brand", "branded", "defense", "defend", "hero", "catalog", "store", "video"
]

PLACEMENT_BUCKET_MAP = {
    "top of search": "top_of_search",
    "top-of-search": "top_of_search",
    "top of search (first page)": "top_of_search",
    "amazon top": "top_of_search",
    "product pages": "product_pages",
    "product page": "product_pages",
    "detail page": "product_pages",
    "detail pages": "product_pages",
    "rest of search": "rest_of_search",
    "other on amazon": "rest_of_search",
    "other": "rest_of_search",
}

DEFAULT_BRAND_TERMS = {
    "brand", "branded", "store", "hero", "defense", "defend", "catalog"
}



def normalize_entity_id(value):
    return str(value or "").strip().replace(".0", "")


import re

def canonicalize_term(term):
    term = str(term or "").lower().strip()
    term = re.sub(r"[^a-z0-9\s]", "", term)
    term = re.sub(r"\s+", " ", term)

    words = []
    for w in term.split():
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        words.append(w)

    return " ".join(words).strip()


def levenshtein_distance(a, b):
    a = str(a or "")
    b = str(b or "")
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def is_semantic_duplicate(term_a, term_b):
    a = canonicalize_term(term_a)
    b = canonicalize_term(term_b)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True

    # word-level fuzzy matching for close brand typos and singular/plural drift
    a_words = a.split()
    b_words = b.split()
    for aw in a_words:
        for bw in b_words:
            if aw == bw:
                continue
            if levenshtein_distance(aw, bw) <= 1:
                return True

    # allow slightly broader fuzzy matching at phrase level for Amazon duplicate behavior
    if levenshtein_distance(a, b) <= 2:
        return True

    return False

def normalize_key(value):
    return " ".join(str(value or "").strip().lower().split())


def normalize_placement_bucket(value):
    raw = normalize_key(value).replace("_", " ").replace("-", " ")
    return PLACEMENT_BUCKET_MAP.get(raw, raw.replace(" ", "_") if raw else "unknown")


def compute_confidence_level(clicks, orders, spend, sales=0.0):
    clicks = float(clicks or 0)
    orders = float(orders or 0)
    spend = float(spend or 0)
    sales = float(sales or 0)
    if orders >= 3 or (sales > 0 and clicks >= 12):
        return "HIGH"
    if clicks >= 10 or spend >= 15:
        return "MEDIUM"
    return "LOW"


def classify_brand_segment(text, campaign_name="", brand_terms=None):
    phrase = f"{campaign_name} {text}".strip().lower()
    if phrase == "":
        return "unknown"

    dynamic_brand_terms = set(DEFAULT_BRAND_TERMS)
    if brand_terms:
        dynamic_brand_terms.update(
            [normalize_key(x) for x in brand_terms if normalize_key(x)]
        )

    competitor_patterns = [
        "vs ", "compare ", "alternative", "similar to", "replacement for",
        "instead of", "better than", "competitor", "compare to"
    ]

    if any(token in phrase for token in competitor_patterns):
        return "competitor"

    if any(term and term in phrase for term in dynamic_brand_terms):
        return "branded"

    return "non_branded"


def classify_match_funnel(match_type, campaign_name="", ad_group_name=""):
    mt = normalize_key(match_type)
    naming = f"{campaign_name} {ad_group_name}".lower()

    if mt in {"exact"}:
        return "capture"
    if mt in {"phrase", "broad"}:
        return "discovery"
    if "auto" in naming or mt in {"close-match", "loose-match", "substitutes", "complements"}:
        return "discovery"
    return "mixed"


def dynamic_step_from_gap(gap_ratio, small_step, large_step):
    if gap_ratio >= 0.35:
        return large_step
    if gap_ratio >= 0.15:
        return (small_step + large_step) / 2.0
    return small_step


def round_to_step(value, step, minimum=None, maximum=None, decimals=2):
    if value in (None, ""):
        return value
    numeric = float(value)
    if step and step > 0:
        numeric = round(numeric / step) * step
    if minimum is not None:
        numeric = max(minimum, numeric)
    if maximum is not None:
        numeric = min(maximum, numeric)
    return round(numeric, decimals)


def round_bid_value(value, max_bid_cap=None):
    return round_to_step(value, step=0.05, minimum=0.02, maximum=max_bid_cap, decimals=2)


def round_budget_value(value, max_budget_cap=None):
    return round_to_step(value, step=5.0, minimum=1.0, maximum=max_budget_cap, decimals=2)


def round_placement_pct(value):
    return int(round_to_step(value, step=5.0, minimum=0.0, maximum=900.0, decimals=0))



def trend_direction(current_value, previous_value, tolerance=0.03):
    current_value = float(current_value or 0)
    previous_value = float(previous_value or 0)
    if previous_value <= 0:
        return "flat"
    delta = (current_value - previous_value) / previous_value
    if delta >= tolerance:
        return "up"
    if delta <= -tolerance:
        return "down"
    return "flat"




class Phase1UploadValidator:
    """Phase 1 upload validation and spend reconciliation.

    This class is intentionally additive. It does not change the existing
    Sponsored Products optimizer logic. It only validates shared / SP / SB / SD
    upload readiness and builds a spend reconciliation view before optimization.
    """

    SP_BULK_SHEETS = ["Sponsored Products Campaigns", "Sponsored Products", "SP Campaigns", "SP", "SP Search Term Report"]
    SB_BULK_SHEETS = ["Sponsored Brands Campaigns", "Sponsored Brands", "SB Campaigns", "SB", "SB Multi Ad Group Campaigns", "SB Search Term Report"]
    SD_BULK_SHEETS = ["Sponsored Display Campaigns", "Sponsored Display", "SD Campaigns", "SD", "RAS Campaigns", "RAS Search Term Report"]

    def __init__(
        self,
        bulk_file=None,
        business_report_file=None,
        sqp_report_file=None,
        margin_report_file=None,
        sp_search_term_file=None,
        sp_targeting_file=None,
        sp_impression_share_file=None,
        sb_search_term_file=None,
        sb_impression_share_file=None,
        sd_targeting_file=None,
    ):
        self.bulk_file = bulk_file
        self.business_report_file = business_report_file
        self.sqp_report_file = sqp_report_file
        self.margin_report_file = margin_report_file  # deprecated for Amazon-only mode
        self.sp_search_term_file = sp_search_term_file
        self.sp_targeting_file = sp_targeting_file
        self.sp_impression_share_file = sp_impression_share_file
        self.sb_search_term_file = sb_search_term_file
        self.sb_impression_share_file = sb_impression_share_file
        self.sd_targeting_file = sd_targeting_file

    def _clone_file_obj(self, file_obj):
        if file_obj is None:
            return None
        if isinstance(file_obj, (str, bytes, os.PathLike)):
            return file_obj
        if isinstance(file_obj, io.BytesIO):
            return io.BytesIO(file_obj.getvalue())
        if hasattr(file_obj, 'getvalue'):
            return io.BytesIO(file_obj.getvalue())
        if hasattr(file_obj, 'read'):
            try:
                pos = file_obj.tell()
            except Exception:
                pos = None
            try:
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                data = file_obj.read()
            finally:
                try:
                    if pos is not None and hasattr(file_obj, 'seek'):
                        file_obj.seek(pos)
                except Exception:
                    pass
            return io.BytesIO(data)
        raise ValueError('Unsupported file object type.')

    def _get_file_extension(self, file_obj, fallback=None):
        if file_obj is None:
            return None
        if isinstance(file_obj, (str, os.PathLike)):
            return os.path.splitext(str(file_obj))[1].lower()
        name = getattr(file_obj, 'name', None)
        if isinstance(name, str) and name.strip():
            return os.path.splitext(name)[1].lower()
        return fallback

    def _load_any(self, file_obj, expected_ext=None, **kwargs):
        if file_obj is None:
            return None
        ext = self._get_file_extension(file_obj, fallback=expected_ext)
        cloned = self._clone_file_obj(file_obj)
        if ext == '.csv':
            return pd.read_csv(cloned, **kwargs)
        if ext in ['.xlsx', '.xls']:
            return pd.read_excel(cloned, engine='openpyxl', **kwargs)
        raise ValueError(f'Unsupported file type: {ext}')

    def _read_bulk_workbook(self):
        if self.bulk_file is None:
            return {}
        cloned = self._clone_file_obj(self.bulk_file)
        return pd.read_excel(cloned, sheet_name=None, engine='openpyxl', dtype=str)

    def _clean_text(self, value):
        return str(value or '').strip()

    def _safe_numeric_series(self, series):
        cleaned = (
            series.fillna('')
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('$', '', regex=False)
            .str.replace('%', '', regex=False)
            .str.replace(r'\(([^\)]+)\)', r'-\1', regex=True)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors='coerce').fillna(0)

    def _sum_spend_from_df(self, df):
        if df is None or getattr(df, 'empty', True):
            return 0.0
        for col in ['Spend', 'Cost', ' spend', 'Amount Spent']:
            if col in df.columns:
                return round(float(self._safe_numeric_series(df[col]).sum()), 2)
        return 0.0

    def _sum_sales_from_df(self, df):
        if df is None or getattr(df, 'empty', True):
            return 0.0
        for col in ['7 Day Total Sales ', 'Sales', '14 Day Total Sales ', 'Attributed Sales', 'Total Sales']:
            if col in df.columns:
                return round(float(self._safe_numeric_series(df[col]).sum()), 2)
        return 0.0

    def _sheet_present(self, workbook, sheet_names):
        return any(name in workbook for name in sheet_names)

    def _status_for(self, ready, missing_required):
        if ready:
            return 'Ready'
        if missing_required:
            return 'Partial'
        return 'Missing'

    def analyze(self):
        workbook = self._read_bulk_workbook()
        bulk_sheet_names = list(workbook.keys())

        ad_types_in_bulk = {
            'SP': self._sheet_present(workbook, self.SP_BULK_SHEETS),
            'SB': self._sheet_present(workbook, self.SB_BULK_SHEETS),
            'SD': self._sheet_present(workbook, self.SD_BULK_SHEETS),
        }

        sp_missing = []
        if not self.bulk_file:
            sp_missing.append('Bulk Sheet')
        if self.sp_search_term_file is None:
            sp_missing.append('SP Search Term Report')
        if self.sp_targeting_file is None:
            sp_missing.append('SP Targeting Report')
        if self.sp_impression_share_file is None:
            sp_missing.append('SP Impression Share Report')

        sb_missing = []
        if not self.bulk_file:
            sb_missing.append('Bulk Sheet')
        if self.sb_search_term_file is None:
            sb_missing.append('SB Search Term Report')

        sd_missing = []
        if not self.bulk_file:
            sd_missing.append('Bulk Sheet')
        if self.sd_targeting_file is None:
            sd_missing.append('SD Targeting Report')

        sp_ready = len(sp_missing) == 0
        sb_ready = len(sb_missing) == 0
        sd_ready = len(sd_missing) == 0

        sp_targeting_df = self._load_any(self.sp_targeting_file, expected_ext='.xlsx') if self.sp_targeting_file is not None else None
        sp_search_df = self._load_any(self.sp_search_term_file, expected_ext='.xlsx') if self.sp_search_term_file is not None else None
        sb_search_df = self._load_any(self.sb_search_term_file, expected_ext='.xlsx') if self.sb_search_term_file is not None else None
        sd_targeting_df = self._load_any(self.sd_targeting_file, expected_ext='.xlsx') if self.sd_targeting_file is not None else None

        sp_spend = self._sum_spend_from_df(sp_targeting_df)
        if sp_spend == 0 and sp_search_df is not None:
            sp_spend = self._sum_spend_from_df(sp_search_df)
        if sp_spend == 0 and 'SP Search Term Report' in workbook:
            sp_spend = self._sum_spend_from_df(workbook['SP Search Term Report'])

        sb_spend = self._sum_spend_from_df(sb_search_df)
        if sb_spend == 0 and 'SB Search Term Report' in workbook:
            sb_spend = self._sum_spend_from_df(workbook['SB Search Term Report'])

        sd_spend = self._sum_spend_from_df(sd_targeting_df)
        if sd_spend == 0 and 'RAS Search Term Report' in workbook:
            sd_spend = self._sum_spend_from_df(workbook['RAS Search Term Report'])

        sp_sales = self._sum_sales_from_df(sp_targeting_df)
        if sp_sales == 0 and sp_search_df is not None:
            sp_sales = self._sum_sales_from_df(sp_search_df)
        if sp_sales == 0 and 'SP Search Term Report' in workbook:
            sp_sales = self._sum_sales_from_df(workbook['SP Search Term Report'])

        sb_sales = self._sum_sales_from_df(sb_search_df)
        if sb_sales == 0 and 'SB Search Term Report' in workbook:
            sb_sales = self._sum_sales_from_df(workbook['SB Search Term Report'])

        sd_sales = self._sum_sales_from_df(sd_targeting_df)
        if sd_sales == 0 and 'RAS Search Term Report' in workbook:
            sd_sales = self._sum_sales_from_df(workbook['RAS Search Term Report'])

        readiness = {
            'SP': {
                'ready': sp_ready,
                'status': self._status_for(sp_ready, sp_missing),
                'missing_required': sp_missing,
                'supported_in_bulk': ad_types_in_bulk['SP'],
                'provided_reports': {
                    'bulk': self.bulk_file is not None,
                    'search_term': self.sp_search_term_file is not None,
                    'targeting': self.sp_targeting_file is not None,
                    'impression_share': self.sp_impression_share_file is not None,
                },
            },
            'SB': {
                'ready': sb_ready,
                'status': self._status_for(sb_ready, sb_missing),
                'missing_required': sb_missing,
                'supported_in_bulk': ad_types_in_bulk['SB'],
                'provided_reports': {
                    'bulk': self.bulk_file is not None,
                    'search_term': self.sb_search_term_file is not None,
                    'impression_share': self.sb_impression_share_file is not None,
                },
            },
            'SD': {
                'ready': sd_ready,
                'status': self._status_for(sd_ready, sd_missing),
                'missing_required': sd_missing,
                'supported_in_bulk': ad_types_in_bulk['SD'],
                'provided_reports': {
                    'bulk': self.bulk_file is not None,
                    'targeting': self.sd_targeting_file is not None,
                },
            },
        }

        return {
            'bulk_sheet_names': bulk_sheet_names,
            'ad_types_in_bulk': ad_types_in_bulk,
            'readiness': readiness,
            'runnable_types': [k for k, v in readiness.items() if v['ready']],
            'spend_summary': {
                'sp_spend': round(sp_spend, 2),
                'sb_spend': round(sb_spend, 2),
                'sd_spend': round(sd_spend, 2),
                'total_spend': round(sp_spend + sb_spend + sd_spend, 2),
                'sp_sales': round(sp_sales, 2),
                'sb_sales': round(sb_sales, 2),
                'sd_sales': round(sd_sales, 2),
                'total_sales': round(sp_sales + sb_sales + sd_sales, 2),
            },
        }

class AdsOptimizerEngine:
    def __init__(
        self,
        bulk_file,
        search_term_file,
        targeting_file,
        impression_share_file,
        business_report_file=None,
        sqp_report_file=None,
        margin_report_file=None,
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
        cooldown_days=7,
        budget_reallocation_pct=0.15,
        enable_placement_weighting=True,
        enable_portfolio_budget_reallocation=True,
        trend_lookback_days=21,
        trend_change_tolerance=0.10,
        brand_terms=None,
        branded_harvest_order_threshold=2,
        branded_scale_roas_floor=0.85,
        branded_negative_multiplier=1.50,
        external_total_ad_spend=None,
        tacos_constrained_override=False,
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
        self.cooldown_days = int(cooldown_days)
        self.budget_reallocation_pct = float(budget_reallocation_pct)
        self.enable_placement_weighting = bool(enable_placement_weighting)
        self.enable_portfolio_budget_reallocation = bool(enable_portfolio_budget_reallocation)
        self.trend_lookback_days = int(trend_lookback_days)
        self.trend_change_tolerance = float(trend_change_tolerance)
        self.brand_terms = [normalize_key(x) for x in (brand_terms or []) if normalize_key(x)]
        self.branded_harvest_order_threshold = int(branded_harvest_order_threshold)
        self.branded_scale_roas_floor = float(branded_scale_roas_floor)
        self.branded_negative_multiplier = float(branded_negative_multiplier)
        self.external_total_ad_spend = (
            float(external_total_ad_spend)
            if external_total_ad_spend not in [None, ""]
            else None
        )
        self.tacos_constrained_override = bool(tacos_constrained_override)
        self.tacos_constrained = False

        self.existing_any_keywords = set()
        self.existing_negative_keywords = set()
        self.keyword_capable_ad_groups = set()
        self.margin_lookup = {}
        self.run_history_cache = pd.DataFrame()
        self.action_history_cache = pd.DataFrame()

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
        workbook = pd.read_excel(
            cloned,
            sheet_name=None,
            engine="openpyxl",
            dtype=str,
        )
        sp_sheet_name = find_matching_sheet_name(
            workbook.keys(),
            SP_BULK_SHEET_CANDIDATES,
        )
        if sp_sheet_name is None:
            raise ValueError(
                f"Could not find a Sponsored Products bulk sheet. Found sheets: {list(workbook.keys())}"
            )
        return workbook[sp_sheet_name]

    def load_sqp_simple_view(self):
        if self.sqp_report_file is None:
            return None

        cloned = self._clone_file_obj(self.sqp_report_file)
        return pd.read_csv(cloned, header=1)

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


    def extract_margin_lookup(self):
        """Deprecated in Amazon-only mode. Margin inputs are ignored."""
        self.margin_lookup = {}
        return {}

    def detect_brand_segment(self, term, campaign_name=""):
        return classify_brand_segment(term, campaign_name, brand_terms=self.brand_terms)

    def detect_funnel_stage(self, match_type, campaign_name="", ad_group_name=""):
        return classify_match_funnel(match_type, campaign_name, ad_group_name)

    def classify_campaign_intent(self, campaign_name="", brand_segment="", funnel_stage=""):
        name = normalize_key(campaign_name)
        brand_segment = normalize_key(brand_segment)
        funnel_stage = normalize_key(funnel_stage)

        if brand_segment == "branded" or any(token in name for token in ["brand", "defense", "protect", "hero"]):
            return "defense"
        if any(token in name for token in ["launch", "rank", "ranking", "boost", "aggressive"]):
            return "rank"
        if any(token in name for token in ["discover", "discovery", "broad", "auto", "prospecting", "research"]):
            return "discovery"
        if any(token in name for token in ["scale", "scaling", "winner", "growth"]):
            return "scale"
        if any(token in name for token in ["efficiency", "profit", "profitable", "tacos", "roas"]):
            return "efficiency"
        if funnel_stage == "capture":
            return "capture"
        return "balanced"

    def get_effective_target(self, row, adjusted_target):
        effective_target = float(adjusted_target or 0)
        brand_segment = str(row.get("brand_segment", "") or "")
        campaign_intent = str(row.get("campaign_intent", "") or "")

        if brand_segment == "branded":
            effective_target = max(effective_target * self.branded_scale_roas_floor, 1.0)
        elif brand_segment == "competitor":
            effective_target = effective_target * 1.10

        if campaign_intent == "defense":
            effective_target = max(effective_target * 0.92, 1.0)
        elif campaign_intent == "discovery":
            effective_target = max(effective_target * 0.95, 1.0)
        elif campaign_intent == "rank":
            effective_target = max(effective_target * 0.90, 1.0)
        elif campaign_intent == "efficiency":
            effective_target = effective_target * 1.05
        elif campaign_intent == "scale":
            effective_target = max(effective_target * 0.97, 1.0)

        return round(float(effective_target), 4)

    def action_clears_execution_threshold(self, action, current_value, new_value):
        action = str(action or "").upper()
        current_value = float(current_value or 0)
        new_value = float(new_value or 0)

        if action in {"INCREASE_BID", "DECREASE_BID"}:
            return abs(new_value - current_value) >= 0.05
        if action in {"INCREASE_BUDGET", "DECREASE_BUDGET"}:
            return abs(new_value - current_value) >= 5.0
        if action == "SET_PLACEMENT_MULTIPLIER":
            return abs(new_value - current_value) >= 5.0
        return True

    def passes_repeat_support_gate(self, row, desired_direction):
        recent_action_count = int(float(row.get("recent_action_count", 0) or 0))
        prior_action_direction = str(row.get("prior_action_direction", "") or "")
        roas_trend = str(row.get("roas_trend", "flat") or "flat")
        order_trend = str(row.get("order_trend", "flat") or "flat")

        if recent_action_count <= 0:
            return True

        if desired_direction == "increase":
            if prior_action_direction == "increase" and not (roas_trend == "up" or order_trend == "up"):
                return False
            if prior_action_direction == "decrease" and roas_trend == "up":
                return False

        if desired_direction == "decrease":
            if prior_action_direction == "decrease" and not (roas_trend == "down" or order_trend == "down"):
                return False
            if prior_action_direction == "increase" and roas_trend == "down":
                return True

        return True

    def adjust_roas_for_margin(self, base_roas, margin_pct):
        """Deprecated in Amazon-only mode. Returns the original ROAS unchanged."""
        return float(base_roas or 0)

    def get_history_signature(self, row, level="target"):
        campaign = normalize_key(row.get("campaign_name", ""))
        ad_group = normalize_key(row.get("ad_group_name", ""))
        target = normalize_key(row.get("target", row.get("search_term", "")))
        match_type = normalize_key(row.get("match_type", ""))
        if level == "campaign":
            return campaign
        return "||".join([campaign, ad_group, target, match_type])

    def annotate_recent_actions(self, df, level="target"):
        out = df.copy()
        out["cooldown_active"] = False
        out["recent_action_count"] = 0
        out["last_action_days_ago"] = np.nan
        out["previous_roas"] = np.nan
        out["previous_clicks"] = np.nan
        out["previous_orders"] = np.nan
        out["previous_score"] = np.nan
        out["prior_action_direction"] = ""
        out["roas_trend"] = "flat"
        out["click_trend"] = "flat"
        out["order_trend"] = "flat"

        history = self.load_action_history()
        if history is None or history.empty or "timestamp" not in history.columns:
            return out

        hist = history.copy()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
        hist = hist.dropna(subset=["timestamp"])
        if hist.empty:
            return out

        latest_ts = hist["timestamp"].max()
        cutoff = latest_ts - pd.Timedelta(days=self.trend_lookback_days)
        hist = hist[hist["timestamp"] >= cutoff].copy()
        if hist.empty:
            return out

        hist["signature"] = hist.apply(lambda r: self.get_history_signature(r, level=level), axis=1)
        hist = hist.sort_values(["signature", "timestamp"])

        last_rows = hist.groupby("signature", as_index=False).tail(1).copy()
        counts = hist.groupby("signature", as_index=False).agg(
            recent_action_count=("signature", "count"),
            last_action_ts=("timestamp", "max"),
        )

        merged_hist = last_rows.merge(counts, on="signature", how="left")

        def infer_prior_direction(row):
            action = str(
                row.get("recommended_action")
                or row.get("campaign_action")
                or row.get("search_term_action")
                or ""
            ).upper()
            if "INCREASE" in action:
                return "increase"
            if "DECREASE" in action or "NEGATIVE" in action:
                return "decrease"
            return "flat"

        merged_hist["prior_action_direction"] = merged_hist.apply(infer_prior_direction, axis=1)
        merged_hist["last_action_days_ago"] = (latest_ts - merged_hist["last_action_ts"]).dt.days

        out["signature"] = out.apply(lambda r: self.get_history_signature(r, level=level), axis=1)

        history_features = merged_hist[
            [
                "signature", "recent_action_count", "last_action_days_ago",
                "roas", "clicks", "orders", "score", "prior_action_direction"
            ]
        ].rename(columns={
            "roas": "previous_roas",
            "clicks": "previous_clicks",
            "orders": "previous_orders",
            "score": "previous_score",
        })

        # Remove default placeholders before merge so pandas does not create _x / _y suffixes.
        overlap_cols = [
            "recent_action_count", "last_action_days_ago", "previous_roas",
            "previous_clicks", "previous_orders", "previous_score", "prior_action_direction"
        ]
        drop_cols = [c for c in overlap_cols if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        out = out.merge(history_features, on="signature", how="left")
        out = ensure_trend_columns(out)

        out["recent_action_count"] = pd.to_numeric(out["recent_action_count"], errors="coerce").fillna(0)
        out["cooldown_active"] = out["recent_action_count"] > 0
        out["roas_trend"] = out.apply(
            lambda r: trend_direction(r.get("roas", 0), r.get("previous_roas", 0), self.trend_change_tolerance),
            axis=1,
        )
        out["click_trend"] = out.apply(
            lambda r: trend_direction(r.get("clicks", 0), r.get("previous_clicks", 0), self.trend_change_tolerance),
            axis=1,
        )
        out["order_trend"] = out.apply(
            lambda r: trend_direction(r.get("orders", 0), r.get("previous_orders", 0), self.trend_change_tolerance),
            axis=1,
        )

        return out


    def compute_action_confidence(self, row):
        return compute_confidence_level(row.get("clicks", 0), row.get("orders", 0), row.get("spend", 0), row.get("sales", 0))

    def build_reason_text(self, row, action, adjusted_target, score=None, extra=None):
        pieces = [
            f"action={action}",
            f"roas={round(float(row.get('roas', 0) or 0), 2)}",
            f"target={round(float(adjusted_target or 0), 2)}",
            f"clicks={int(float(row.get('clicks', 0) or 0))}",
            f"orders={round(float(row.get('orders', 0) or 0), 2)}",
            f"spend={round(float(row.get('spend', 0) or 0), 2)}",
        ]
        if score is not None:
            pieces.append(f"score={round(float(score), 3)}")
        if row.get("brand_segment"):
            pieces.append(f"segment={row.get('brand_segment')}")
        if row.get("funnel_stage"):
            pieces.append(f"stage={row.get('funnel_stage')}")
        if row.get("placement_bucket"):
            pieces.append(f"placement={row.get('placement_bucket')}")
        if extra:
            pieces.append(str(extra))
        return " | ".join(pieces)

    def calculate_weighted_efficiency_score(self, row, adjusted_target):
        roas = float(row.get("roas", 0) or 0)
        clicks = float(row.get("clicks", 0) or 0)
        orders = float(row.get("orders", 0) or 0)
        spend = float(row.get("spend", 0) or 0)
        impression_share = float(row.get("impression_share_pct", 0) or 0)
        cvr = float(row.get("cvr", 0) or 0)
        ctr = float(row.get("ctr", 0) or 0)

        brand_segment = str(row.get("brand_segment", "") or "")
        funnel_stage = str(row.get("funnel_stage", "") or "")
        placement_bucket = str(row.get("placement_bucket", "") or "")
        roas_trend = str(row.get("roas_trend", "flat") or "flat")
        order_trend = str(row.get("order_trend", "flat") or "flat")
        campaign_intent = str(row.get("campaign_intent", "") or "")

        effective_target = self.get_effective_target(row, adjusted_target)
        roas_ratio = (roas / effective_target) if effective_target > 0 else 0

        score = 0.0
        score += min(roas_ratio, 2.0) * 0.34
        score += min(orders / max(self.min_orders_for_scaling, 1), 2.0) * 0.18
        score += min(clicks / max(self.min_clicks, 1), 2.0) * 0.08
        score += min(cvr * 10, 1.5) * 0.14
        score += min(ctr * 20, 1.0) * 0.05

        if impression_share > 0 and impression_share < 20:
            score += 0.08

        if placement_bucket == "top_of_search" and roas_ratio >= 1:
            score += 0.10
        elif placement_bucket == "product_pages" and roas_ratio < 1:
            score -= 0.08
        elif placement_bucket == "rest_of_search" and roas_ratio >= 1:
            score += 0.03

        if brand_segment == "branded":
            score += 0.08
        elif brand_segment == "competitor":
            score -= 0.08

        if campaign_intent in {"scale", "rank"} and roas_ratio >= 1:
            score += 0.05
        if campaign_intent == "efficiency" and roas_ratio < 1:
            score -= 0.06
        if campaign_intent == "defense" and brand_segment == "branded" and roas_ratio >= 0.9:
            score += 0.04

        if funnel_stage == "capture":
            score += 0.04

        if roas_trend == "up":
            score += 0.08
        elif roas_trend == "down":
            score -= 0.08

        if order_trend == "up":
            score += 0.05
        elif order_trend == "down":
            score -= 0.05

        if spend >= 15 and orders == 0:
            score -= 0.40
        if clicks >= self.zero_order_click_threshold and orders == 0:
            score -= 0.25
        if row.get("cooldown_active"):
            score -= 0.10
        if int(float(row.get("recent_action_count", 0) or 0)) >= 2 and roas_trend == "flat":
            score -= 0.05

        return round(score, 4)


    def determine_dynamic_bid_change(self, row, score, adjusted_target):
        current_bid = float(row.get("current_bid", 0) or 0)
        if current_bid <= 0:
            return "NO_ACTION", current_bid, 0.0

        roas = float(row.get("roas", 0) or 0)
        orders = float(row.get("orders", 0) or 0)
        clicks = float(row.get("clicks", 0) or 0)

        brand_segment = str(row.get("brand_segment", "") or "")
        placement_bucket = str(row.get("placement_bucket", "") or "")
        campaign_intent = str(row.get("campaign_intent", "") or "")
        roas_trend = str(row.get("roas_trend", "flat") or "flat")
        order_trend = str(row.get("order_trend", "flat") or "flat")

        effective_target = self.get_effective_target(row, adjusted_target)
        gap_ratio = abs((roas - effective_target) / effective_target) if effective_target > 0 else 0
        increase_step = dynamic_step_from_gap(gap_ratio, min(0.04, self.max_bid_up), self.max_bid_up)
        decrease_step = dynamic_step_from_gap(gap_ratio, min(0.06, self.max_bid_down), self.max_bid_down)

        if self.enable_placement_weighting and placement_bucket == "top_of_search" and roas >= effective_target:
            increase_step = min(self.max_bid_up, increase_step + 0.04)
        if self.enable_placement_weighting and placement_bucket == "product_pages" and roas < effective_target:
            decrease_step = min(self.max_bid_down, decrease_step + 0.03)
        if self.enable_placement_weighting and placement_bucket == "rest_of_search" and roas >= effective_target * 1.10:
            increase_step = min(self.max_bid_up, increase_step + 0.01)

        if campaign_intent in {"scale", "rank"} and roas >= effective_target:
            increase_step = min(self.max_bid_up, increase_step + 0.02)
        if campaign_intent == "efficiency":
            decrease_step = min(self.max_bid_down, decrease_step + 0.02)
            increase_step = max(0.0, increase_step - 0.02)
        if campaign_intent == "defense" and brand_segment == "branded":
            decrease_step = min(decrease_step, 0.06)

        if self.should_zero_order_decrease_bid() and clicks >= self.zero_order_click_threshold and orders == 0:
            if brand_segment == "branded" and campaign_intent == "defense":
                return "NO_ACTION", current_bid, 0.0
            if not self.passes_repeat_support_gate(row, "decrease"):
                return "NO_ACTION", current_bid, 0.0
            new_bid = round_bid_value(current_bid * (1 - max(decrease_step, 0.08)), self.max_bid_cap)
            if not self.action_clears_execution_threshold("DECREASE_BID", current_bid, new_bid):
                return "NO_ACTION", current_bid, 0.0
            return "DECREASE_BID", new_bid, max(decrease_step, 0.08)

        if roas < effective_target and clicks >= self.min_clicks:
            if not self.passes_repeat_support_gate(row, "decrease"):
                return "NO_ACTION", current_bid, 0.0
            new_bid = round_bid_value(current_bid * (1 - decrease_step), self.max_bid_cap)
            if not self.action_clears_execution_threshold("DECREASE_BID", current_bid, new_bid):
                return "NO_ACTION", current_bid, 0.0
            return "DECREASE_BID", new_bid, decrease_step

        if (
            not self.build_budget_pacing_status().get("over_pace")
            and score >= 0.98
            and orders >= self.min_orders_for_scaling
            and roas >= effective_target * 1.05
            and (roas_trend == "up" or order_trend == "up" or campaign_intent in {"scale", "rank", "defense"})
            and self.passes_repeat_support_gate(row, "increase")
        ):
            new_bid = round_bid_value(current_bid * (1 + increase_step), self.max_bid_cap)
            if not self.action_clears_execution_threshold("INCREASE_BID", current_bid, new_bid):
                return "NO_ACTION", current_bid, 0.0
            return "INCREASE_BID", new_bid, increase_step

        return "NO_ACTION", current_bid, 0.0


    def determine_dynamic_budget_change(self, row, adjusted_target):
        current_budget = float(row.get("daily_budget", 0) or 0)
        if current_budget <= 0:
            return "NO_ACTION", current_budget, 0.0

        roas = float(row.get("roas", 0) or 0)
        orders = float(row.get("orders", 0) or 0)
        clicks = float(row.get("clicks", 0) or 0)
        share = float(row.get("avg_impression_share_pct", 0) or 0)
        brand_segment = str(row.get("brand_segment", "") or "")
        campaign_intent = str(row.get("campaign_intent", "") or "")
        roas_trend = str(row.get("roas_trend", "flat") or "flat")
        order_trend = str(row.get("order_trend", "flat") or "flat")

        effective_target = self.get_effective_target(row, adjusted_target)
        gap_ratio = abs((roas - effective_target) / effective_target) if effective_target > 0 else 0
        up_step = dynamic_step_from_gap(gap_ratio, min(0.04, self.budget_up_pct), self.budget_up_pct)
        down_step = dynamic_step_from_gap(gap_ratio, min(0.05, self.budget_down_pct), self.budget_down_pct)

        if self.build_budget_pacing_status().get("over_pace"):
            up_step = 0.0

        if campaign_intent in {"scale", "rank"} and roas >= effective_target:
            up_step = min(self.budget_up_pct, up_step + 0.03)
        if campaign_intent == "efficiency":
            down_step = min(self.budget_down_pct, down_step + 0.03)
            up_step = max(0.0, up_step - 0.02)
        if campaign_intent == "defense" and brand_segment == "branded":
            down_step = min(down_step, 0.08)

        if roas >= effective_target * 1.10 and orders >= self.min_orders_for_scaling and share < 25 and up_step > 0:
            if self.passes_repeat_support_gate(row, "increase") and (roas_trend == "up" or order_trend == "up" or campaign_intent in {"scale", "rank", "defense"}):
                new_budget = round_budget_value(current_budget * (1 + up_step), self.max_budget_cap)
                if not self.action_clears_execution_threshold("INCREASE_BUDGET", current_budget, new_budget):
                    return "NO_ACTION", current_budget, 0.0
                return "INCREASE_BUDGET", new_budget, up_step

        if (roas < effective_target and clicks >= max(self.min_clicks * 3, 25)) or (orders == 0 and float(row.get("spend", 0) or 0) >= 20):
            if brand_segment == "branded" and campaign_intent == "defense" and roas_trend == "up":
                return "NO_ACTION", current_budget, 0.0
            if not self.passes_repeat_support_gate(row, "decrease"):
                return "NO_ACTION", current_budget, 0.0
            new_budget = round_budget_value(current_budget * (1 - down_step), self.max_budget_cap)
            if not self.action_clears_execution_threshold("DECREASE_BUDGET", current_budget, new_budget):
                return "NO_ACTION", current_budget, 0.0
            return "DECREASE_BUDGET", new_budget, down_step

        return "NO_ACTION", current_budget, 0.0


    # -----------------------------
    # CAMPAIGN HEALTH DASHBOARD
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
    # SQP NORMALIZATION + OPPORTUNITIES
    # -----------------------------
    def normalize_sqp(self):
        if self.sqp_df is None or self.sqp_df.empty:
            return pd.DataFrame()

        df = self.sqp_df.copy()

        normalized = pd.DataFrame()
        normalized["search_query"] = self.clean_text(df["Search Query"])
        normalized["search_query_score"] = self.safe_numeric(df["Search Query Score"])
        normalized["search_query_volume"] = self.safe_numeric(df["Search Query Volume"])
        normalized["impressions_total_count"] = self.safe_numeric(df["Impressions: Total Count"])
        normalized["impressions_brand_share_pct"] = self.clean_percent_series(df["Impressions: Brand Share %"])
        normalized["clicks_total_count"] = self.safe_numeric(df["Clicks: Total Count"])
        normalized["clicks_click_rate_pct"] = self.clean_percent_series(df["Clicks: Click Rate %"])
        normalized["clicks_brand_share_pct"] = self.clean_percent_series(df["Clicks: Brand Share %"])
        normalized["cart_adds_total_count"] = self.safe_numeric(df["Cart Adds: Total Count"])
        normalized["cart_adds_brand_share_pct"] = self.clean_percent_series(df["Cart Adds: Brand Share %"])
        normalized["purchases_total_count"] = self.safe_numeric(df["Purchases: Total Count"])
        normalized["purchase_rate_pct"] = self.clean_percent_series(df["Purchases: Purchase Rate %"])
        normalized["purchases_brand_share_pct"] = self.clean_percent_series(df["Purchases: Brand Share %"])

        reporting_date_col = self.get_optional_column(df, ["Reporting Date"])
        normalized["reporting_date"] = self.clean_text(df[reporting_date_col]) if reporting_date_col else ""

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
                "harvest_overlap": 0,
            }

        sqp = sqp_df.copy()

        search_term_set = set()
        if search_terms_df is not None and not search_terms_df.empty:
            search_term_set = set(
                search_terms_df["search_term"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
                .tolist()
            )

        sqp["search_query_key"] = (
            sqp["search_query"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )

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
            [
                "Test as Exact / Phrase",
                "Prioritize existing query for harvest or campaign expansion",
                "Monitor and test selectively",
            ],
            default="No immediate action",
        )

        sqp["opportunity_score"] = (
            (sqp["search_query_score"] * 0.35)
            + (sqp["search_query_volume"].clip(upper=20000) / 20000 * 100 * 0.25)
            + (sqp["purchase_rate_pct"].clip(upper=15) / 15 * 100 * 0.25)
            + ((100 - sqp["purchases_brand_share_pct"].clip(upper=100)) * 0.15)
        ).round(2)

        opportunities = sqp.copy()
        opportunities["opportunity_tier"] = pd.Categorical(
            opportunities["opportunity_tier"],
            categories=["High Opportunity", "Monitor", "Low Priority"],
            ordered=True,
        )

        opportunities = opportunities.sort_values(
            by=["opportunity_tier", "opportunity_score", "search_query_volume"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

        summary = {
            "uploaded": True,
            "total_queries": int(len(opportunities)),
            "high_opportunity": int((opportunities["opportunity_tier"] == "High Opportunity").sum()),
            "monitor": int((opportunities["opportunity_tier"] == "Monitor").sum()),
            "low_priority": int((opportunities["opportunity_tier"] == "Low Priority").sum()),
            "harvest_overlap": int(
                (
                    (opportunities["opportunity_tier"] == "High Opportunity")
                    & (opportunities["in_search_term_report"])
                ).sum()
            ),
        }

        return opportunities, summary

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
        sqp_summary=None,
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

        if sqp_summary and sqp_summary.get("uploaded"):
            high_opp = int(sqp_summary.get("high_opportunity", 0))
            overlap = int(sqp_summary.get("harvest_overlap", 0))

            if high_opp > 0:
                warnings.append(
                    f"{high_opp} high-opportunity SQP queries were identified from the prior-month Simple View report."
                )
                suggestions.append(
                    "Use SQP opportunities to guide new keyword expansion and prioritize harvest reviews."
                )

            if overlap > 0:
                warnings.append(
                    f"{overlap} high-opportunity SQP queries also appear in the search term report."
                )
                suggestions.append(
                    "These overlap queries may deserve faster campaign buildout or harvest prioritization."
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
            "warnings": warnings[:8],
            "suggestions": suggestions[:8],
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
            "high_confidence_actions": int((bid_recommendations.get("confidence", pd.Series(dtype=str)) == "HIGH").sum())
            + int((search_term_actions.get("confidence", pd.Series(dtype=str)) == "HIGH").sum())
            + int((campaign_budget_actions.get("confidence", pd.Series(dtype=str)) == "HIGH").sum()),
            "low_confidence_actions": int((bid_recommendations.get("confidence", pd.Series(dtype=str)) == "LOW").sum())
            + int((search_term_actions.get("confidence", pd.Series(dtype=str)) == "LOW").sum())
            + int((campaign_budget_actions.get("confidence", pd.Series(dtype=str)) == "LOW").sum()),
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
            updated = safe_concat_frames([existing, new_row_df], ignore_index=True)
        else:
            updated = new_row_df

        updated.to_csv(history_path, index=False)

    def load_run_history(self):
        history_path = "run_history.csv"
        if os.path.exists(history_path):
            return pd.read_csv(history_path)
        return pd.DataFrame()


    def load_action_history(self):
        history_path = "action_history.csv"
        if os.path.exists(history_path):
            return pd.read_csv(history_path)
        return pd.DataFrame()

    def save_action_history(self, action_df: pd.DataFrame, entity_level: str):
        if action_df is None or action_df.empty:
            return

        history_path = "action_history.csv"
        export = action_df.copy()
        export["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        export["entity_level"] = entity_level

        keep_cols = [
            "timestamp", "entity_level", "campaign_name", "ad_group_name", "target", "search_term",
            "match_type", "recommended_action", "campaign_action", "search_term_action",
            "recommended_bid", "recommended_daily_budget", "score", "roas", "clicks", "orders",
            "spend", "sales", "brand_segment", "placement_bucket"
        ]
        keep_cols = [c for c in keep_cols if c in export.columns]
        export = export[keep_cols].copy()

        if os.path.exists(history_path):
            existing = pd.read_csv(history_path)
            export = pd.concat([existing, export], ignore_index=True)

        export.to_csv(history_path, index=False)

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
        sqp = self.normalize_sqp()

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
        bid_recommendations = ensure_trend_columns(ensure_score_column(bid_recommendations))

        search_term_actions = self.build_search_term_actions(
            search_terms,
            adjusted_min_roas,
        )
        search_term_actions = ensure_trend_columns(ensure_score_column(search_term_actions))

        campaign_budget_actions = self.build_campaign_budget_actions(
            targeting_with_share,
            bulk_campaigns,
            adjusted_min_roas,
        )
        campaign_budget_actions = ensure_trend_columns(ensure_score_column(campaign_budget_actions))

        campaign_health_dashboard = self.build_campaign_health_dashboard(
            targeting_with_share,
            adjusted_min_roas,
        )
        campaign_health_dashboard = ensure_trend_columns(ensure_score_column(campaign_health_dashboard))

        sqp_opportunities, sqp_summary = self.build_sqp_opportunities(
            sqp_df=sqp,
            search_terms_df=search_terms,
        )

        smart = self.build_smart_warnings(
            targeting_with_share_df=targeting_with_share,
            search_terms_df=search_terms,
            campaign_health_df=campaign_health_dashboard,
            account_health=account_health,
            adjusted_min_roas=adjusted_min_roas,
            sqp_summary=sqp_summary,
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

        bid_recommendations = ensure_trend_columns(ensure_score_column(bid_recommendations))
        search_term_actions = ensure_trend_columns(ensure_score_column(search_term_actions))
        campaign_budget_actions = ensure_trend_columns(ensure_score_column(campaign_budget_actions))
        top_opportunities = ensure_trend_columns(ensure_score_column(top_opportunities)) if 'top_opportunities' in locals() else pd.DataFrame()

        return {
            "account_health": account_health,
            "account_summary": account_summary,
            "campaign_health_dashboard": campaign_health_dashboard,
            "smart_warnings": smart["warnings"],
            "optimization_suggestions": smart["suggestions"],
            "pre_run_preview": pre_run_preview,
            "sqp_opportunities": sqp_opportunities,
            "sqp_summary": sqp_summary,
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
        self.margin_df = None
        self.sqp_df = self.load_sqp_simple_view() if self.sqp_report_file is not None else None
        self.extract_margin_lookup()

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
        placement_col = self.get_optional_column(df, ["Placement", "Placement Type"])
        normalized["placement_bucket"] = (
            df[placement_col].map(normalize_placement_bucket)
            if placement_col else "unknown"
        )

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
        normalized["brand_segment"] = normalized.apply(lambda r: self.detect_brand_segment(r["search_term"], r["campaign_name"]), axis=1)
        normalized["funnel_stage"] = normalized.apply(lambda r: self.detect_funnel_stage(r["match_type"], r["campaign_name"], r["ad_group_name"]), axis=1)
        normalized["confidence"] = normalized.apply(lambda r: self.compute_action_confidence(r), axis=1)

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
        placement_col = self.get_optional_column(df, ["Placement", "Placement Type"])
        normalized["placement_bucket"] = (
            df[placement_col].map(normalize_placement_bucket)
            if placement_col else "unknown"
        )

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
        normalized["brand_segment"] = normalized.apply(lambda r: self.detect_brand_segment(r["target"], r["campaign_name"]), axis=1)
        normalized["funnel_stage"] = normalized.apply(lambda r: self.detect_funnel_stage(r["match_type"], r["campaign_name"], r["ad_group_name"]), axis=1)
        normalized["confidence"] = normalized.apply(lambda r: self.compute_action_confidence(r), axis=1)

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
        placement_col = self.get_optional_column(df, ["Placement", "Placement Type"])
        normalized["placement_bucket"] = (
            df[placement_col].map(normalize_placement_bucket)
            if placement_col else "unknown"
        )

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
        portfolio_name_col = self.get_optional_column(df, ["Portfolio Name", "Portfolio"])
        portfolio_id_col = self.get_optional_column(df, ["Portfolio ID"])

        rows = df[df[entity_col] == "Campaign"].copy()

        normalized = pd.DataFrame()
        normalized["campaign_name"] = self.combine_preferred_columns(
            rows, ["Campaign Name"], ["Campaign Name (Informational only)"], "campaign name"
        )
        normalized["campaign_id"] = self.clean_text(rows[campaign_id_col]) if campaign_id_col else ""
        normalized["daily_budget"] = self.safe_numeric(rows[budget_col]) if budget_col else 0
        normalized["portfolio_name"] = self.clean_text(rows[portfolio_name_col]) if portfolio_name_col else ""
        normalized["portfolio_id"] = self.clean_text(rows[portfolio_id_col]) if portfolio_id_col else ""

        normalized["portfolio_key"] = np.where(
            normalized["portfolio_id"].astype(str).str.strip() != "",
            normalized["portfolio_id"].astype(str).str.strip(),
            normalized["portfolio_name"].astype(str).str.strip().str.lower()
        )

        return normalized

    # -----------------------------
    # BUSINESS REPORT / TACOS
    # -----------------------------
    # BUSINESS REPORT / TACOS
    # -----------------------------
    def build_business_sales_total(self):
        """Build total business sales for TACOS using Seller Central Business Report data."""
        if self.business_df is None or not self.enable_tacos_control:
            return None

        df = self.business_df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        def _clean_money(series):
            return pd.to_numeric(
                series.astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("(", "-", regex=False)
                .str.replace(")", "", regex=False)
                .str.strip(),
                errors="coerce",
            ).fillna(0)

        exact_ordered_cols = [
            c for c in df.columns
            if str(c).strip().lower() == "ordered product sales"
        ]
        if exact_ordered_cols:
            total_sales = sum(float(_clean_money(df[col]).sum()) for col in exact_ordered_cols)
            return round(total_sales, 2)

        ordered_component_cols = [
            c for c in df.columns
            if str(c).strip().lower() in {
                "ordered product sales - b2c",
                "ordered product sales - b2b",
            }
        ]
        if ordered_component_cols:
            total_sales = sum(float(_clean_money(df[col]).sum()) for col in ordered_component_cols)
            return round(total_sales, 2)

        fallback_cols = [
            c for c in df.columns
            if str(c).strip().lower() in {"total sales", "sales"}
        ]
        if fallback_cols:
            return round(float(_clean_money(df[fallback_cols[0]]).sum()), 2)

        return None

    # -----------------------------
    # ACCOUNT HEALTH
    # -----------------------------
    def build_account_health(self, targeting_with_share_df):
        df = targeting_with_share_df.copy()

        channel_spend = df["spend"].sum()
        total_sales = df["sales"].sum()

        account_roas = total_sales / channel_spend if channel_spend > 0 else 0
        waste_spend = df.loc[df["orders"] == 0, "spend"].sum()
        waste_spend_pct = waste_spend / channel_spend if channel_spend > 0 else 0

        effective_total_ad_spend = (
            self.external_total_ad_spend
            if self.external_total_ad_spend is not None and self.external_total_ad_spend > 0
            else channel_spend
        )

        business_total_sales = self.build_business_sales_total()
        tacos = None
        tacos_status = "not_used"

        if business_total_sales is not None and business_total_sales > 0:
            tacos = effective_total_ad_spend / business_total_sales
            tacos_status = "within_target" if tacos <= self.max_tacos_target else "above_target"

        status = "healthy"
        adjusted_min_roas = self.min_roas

        if account_roas < self.min_roas:
            status = "under_target"
            adjusted_min_roas = self.min_roas * self.account_health_tighten_multiplier
        elif account_roas > self.min_roas * 1.2:
            status = "above_target"
            adjusted_min_roas = self.min_roas * 0.95

        if self.tacos_constrained_override or (tacos is not None and tacos > self.max_tacos_target):
            status = "tacos_constrained"
            tacos_status = "above_target"
            adjusted_min_roas = max(adjusted_min_roas, self.min_roas * 1.15)

        self.tacos_constrained = tacos_status == "above_target"
        commercial_efficiency_mode = bool(self.business_df is not None)

        return {
            "account_roas": round(account_roas, 2),
            "waste_spend": round(waste_spend, 2),
            "waste_spend_pct": round(waste_spend_pct * 100, 2),
            "health_status": status,
            "adjusted_min_roas": round(adjusted_min_roas, 2),
            "tacos_pct": round(tacos * 100, 2) if tacos is not None else None,
            "tacos_status": tacos_status,
            "commercial_efficiency_mode": commercial_efficiency_mode,
            "channel_spend": round(channel_spend, 2),
            "effective_total_ad_spend": round(effective_total_ad_spend, 2),
            "business_total_sales": round(business_total_sales, 2) if business_total_sales is not None else None,
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
            + "||"
            + targeting["placement_bucket"].astype(str).str.lower().str.strip()
        )

        impression["join_key"] = (
            impression["campaign_name"].str.lower().str.strip()
            + "||"
            + impression["ad_group_name"].str.lower().str.strip()
            + "||"
            + impression["target"].str.lower().str.strip()
            + "||"
            + impression["match_type"].str.lower().str.strip()
            + "||"
            + impression["placement_bucket"].astype(str).str.lower().str.strip()
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
        perf = self.annotate_recent_actions(targeting_with_share_df.copy(), level="target")
        perf["campaign_intent"] = perf.apply(
            lambda r: self.classify_campaign_intent(r.get("campaign_name", ""), r.get("brand_segment", ""), r.get("funnel_stage", "")),
            axis=1,
        )

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

        actions = []
        recommended_bids = []
        confidence = []
        scores = []
        action_pct = []
        reason = []

        for _, row in recs.iterrows():
            action = "NO_ACTION"
            new_bid = row["current_bid"]
            score = self.calculate_weighted_efficiency_score(row, adjusted_min_roas)
            pct = 0.0

            if row["current_bid"] <= 0 or not self.enable_bid_updates:
                reason.append(self.build_reason_text(row, action, adjusted_min_roas, score=score, extra="bid updates disabled or missing bid"))
                actions.append(action)
                recommended_bids.append(new_bid)
                confidence.append(self.compute_action_confidence(row))
                scores.append(score)
                action_pct.append(pct)
                continue

            action, new_bid, pct = self.determine_dynamic_bid_change(row, score, adjusted_min_roas)
            extra = f"intent={row.get('campaign_intent', 'balanced')}"
            if row.get("cooldown_active") and action == "INCREASE_BID":
                extra += " | cooldown moderation applied"
            if action != "NO_ACTION" and not self.action_clears_execution_threshold(action, row["current_bid"], new_bid):
                action = "NO_ACTION"
                new_bid = row["current_bid"]
                pct = 0.0
                extra += " | noop_suppressed"

            reason.append(self.build_reason_text(row, action, adjusted_min_roas, score=score, extra=extra))
            actions.append(action)
            recommended_bids.append(new_bid)
            confidence.append(self.compute_action_confidence(row))
            scores.append(score)
            action_pct.append(round(pct * 100, 2))

        recs["recommended_action"] = actions
        recs["recommended_bid"] = recommended_bids
        recs["confidence"] = confidence
        recs["score"] = scores
        recs["change_pct"] = action_pct
        recs["reason"] = reason
        recs = recs[
            (recs["recommended_action"] == "NO_ACTION")
            | (pd.to_numeric(recs["recommended_bid"], errors="coerce").fillna(0) != pd.to_numeric(recs["current_bid"], errors="coerce").fillna(0))
        ].copy()
        return recs

    # -----------------------------
    # SEARCH TERM ACTIONS
    # -----------------------------
    def build_search_term_actions(self, search_terms_df, adjusted_min_roas):
        df = self.annotate_recent_actions(search_terms_df.copy(), level="target")
        df["campaign_intent"] = df.apply(
            lambda r: self.classify_campaign_intent(r.get("campaign_name", ""), r.get("brand_segment", ""), r.get("funnel_stage", "")),
            axis=1,
        )

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
        confidence = []
        reasons = []

        for _, row in df.iterrows():
            action = "NO_ACTION"
            rec_bid = round_bid_value(max(float(row.get("cpc", 0) or 0) * 1.10, 0.20), self.max_bid_cap)

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
            keyword_exists_key = campaign_name_l.strip() + "||" + ad_group_name_l.strip() + "||" + normalized_term
            negative_key = campaign_name_l.strip() + "||" + ad_group_name_l.strip() + "||" + normalized_term + "||negative phrase"
            brand_segment = row.get("brand_segment", self.detect_brand_segment(normalized_term, campaign_name))
            campaign_intent = str(row.get("campaign_intent", "") or "")
            roas_trend = str(row.get("roas_trend", "flat") or "flat")
            order_trend = str(row.get("order_trend", "flat") or "flat")

            harvest_threshold_orders = self.branded_harvest_order_threshold if brand_segment == "branded" else 4
            if campaign_intent in {"discovery", "rank"}:
                harvest_threshold_orders = max(2, harvest_threshold_orders - 1)
            if campaign_intent == "efficiency":
                harvest_threshold_orders = max(harvest_threshold_orders, 4)

            negative_click_floor = max(
                self.zero_order_click_threshold,
                int(self.zero_order_click_threshold * self.branded_negative_multiplier) if brand_segment == "branded"
                else 16 if brand_segment == "competitor"
                else 20
            )

            if (
                self.enable_negative_keywords
                and self.should_zero_order_negate()
                and row["clicks"] >= negative_click_floor
                and row["orders"] == 0
                and row["sales"] == 0
                and normalized_term != ""
                and negative_key not in self.existing_negative_keywords
                and len(normalized_term.split()) >= 2
                and brand_segment not in {"branded"}
                and roas_trend != "up"
                and self.passes_repeat_support_gate(row, "decrease")
            ):
                action = "ADD_NEGATIVE_PHRASE"

            elif (
                self.enable_search_harvesting
                and not is_auto_ad_group
                and ad_group_key in self.keyword_capable_ad_groups
                and row["orders"] >= harvest_threshold_orders
                and row["clicks"] >= 5
                and row["roas"] >= max(self.get_effective_target(row, adjusted_min_roas), self.min_roas * (self.branded_scale_roas_floor if brand_segment == "branded" else 1.0))
                and match_type != "exact"
                and normalized_term != ""
                and keyword_exists_key not in self.existing_any_keywords
                and not row.get("cooldown_active", False)
                and (order_trend == "up" or roas_trend in {"up", "flat"} or campaign_intent in {"discovery", "rank", "defense"})
            ):
                action = "HARVEST_TO_EXACT"

            actions.append(action)
            recommended_bids.append(rec_bid)
            confidence.append(self.compute_action_confidence(row))
            reasons.append(self.build_reason_text(row, action, adjusted_min_roas, extra=f"search_term={normalized_term} | intent={campaign_intent}"))

        df["search_term_action"] = actions
        df["recommended_bid"] = recommended_bids
        df["confidence"] = confidence
        df["reason"] = reasons
        return df

    # -----------------------------
    # CAMPAIGN BUDGET ACTIONS
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
        campaign_perf["brand_segment"] = campaign_perf["campaign_name"].apply(lambda x: self.detect_brand_segment("", x))
        campaign_perf["campaign_intent"] = campaign_perf.apply(
            lambda r: self.classify_campaign_intent(r.get("campaign_name", ""), r.get("brand_segment", ""), "capture"),
            axis=1,
        )
        campaign_perf = self.annotate_recent_actions(campaign_perf, level="campaign")

        recs = campaign_perf.merge(
            bulk_campaigns_df,
            on="campaign_name",
            how="left",
        )

        actions = []
        recommended_budgets = []
        confidence = []
        reasons = []
        change_pct = []

        for _, row in recs.iterrows():
            action, new_budget, pct = self.determine_dynamic_budget_change(row, adjusted_min_roas)
            if action != "NO_ACTION" and not self.action_clears_execution_threshold(action, row.get("daily_budget", 0), new_budget):
                action = "NO_ACTION"
                new_budget = row.get("daily_budget", 0)
                pct = 0.0
            actions.append(action)
            recommended_budgets.append(new_budget)
            confidence.append(self.compute_action_confidence(row))
            reasons.append(self.build_reason_text(row, action, adjusted_min_roas, extra=f"campaign_budget | intent={row.get('campaign_intent', 'balanced')}"))
            change_pct.append(round(pct * 100, 2))

        recs["campaign_action"] = actions
        recs["recommended_daily_budget"] = recommended_budgets
        recs["confidence"] = confidence
        recs["reason"] = reasons
        recs["change_pct"] = change_pct
        recs = recs[
            (recs["campaign_action"] == "NO_ACTION")
            | (pd.to_numeric(recs["recommended_daily_budget"], errors="coerce").fillna(0) != pd.to_numeric(recs["daily_budget"], errors="coerce").fillna(0))
        ].copy()
        return recs

    def build_portfolio_budget_reallocation_plan(self, campaign_budget_actions_df):
        if (
            campaign_budget_actions_df is None
            or campaign_budget_actions_df.empty
            or not self.enable_portfolio_budget_reallocation
        ):
            return pd.DataFrame()

        df = campaign_budget_actions_df.copy()
        if "portfolio_key" not in df.columns:
            df["portfolio_key"] = ""

        plan_rows = []

        for portfolio_key, group in df.groupby("portfolio_key", dropna=False):
            if str(portfolio_key).strip() == "":
                continue

            donors = group[group["campaign_action"] == "DECREASE_BUDGET"].copy()
            receivers = group[group["campaign_action"] == "INCREASE_BUDGET"].copy()

            if donors.empty or receivers.empty:
                continue

            donors["available_delta"] = (pd.to_numeric(donors["daily_budget"], errors="coerce").fillna(0) - pd.to_numeric(donors["recommended_daily_budget"], errors="coerce").fillna(0)).clip(lower=0)
            receivers["needed_delta"] = (pd.to_numeric(receivers["recommended_daily_budget"], errors="coerce").fillna(0) - pd.to_numeric(receivers["daily_budget"], errors="coerce").fillna(0)).clip(lower=0)

            total_available = float(donors["available_delta"].sum())
            total_needed = float(receivers["needed_delta"].sum())
            if total_available < 5 or total_needed < 5:
                continue

            transferable = min(total_available, total_needed)
            transferable = round_budget_value(transferable, None)

            donors = donors.sort_values(["score", "roas"], ascending=[True, True]).copy()
            receivers = receivers.sort_values(["score", "roas"], ascending=[False, False]).copy()

            donor_rows = []
            remaining_transfer = transferable
            for _, donor in donors.iterrows():
                if remaining_transfer < 5:
                    break
                donor_take = min(float(donor["available_delta"]), remaining_transfer)
                donor_take = round_budget_value(donor_take, None)
                if donor_take < 5:
                    continue
                donor_rows.append((donor, donor_take))
                remaining_transfer -= donor_take

            receiver_rows = []
            remaining_transfer = transferable
            for _, receiver in receivers.iterrows():
                if remaining_transfer < 5:
                    break
                receiver_give = min(float(receiver["needed_delta"]), remaining_transfer)
                receiver_give = round_budget_value(receiver_give, None)
                if receiver_give < 5:
                    continue
                receiver_rows.append((receiver, receiver_give))
                remaining_transfer -= receiver_give

            for donor, donor_take in donor_rows:
                new_budget = round_budget_value(float(donor.get("daily_budget", 0)) - donor_take, self.max_budget_cap)
                if not self.action_clears_execution_threshold("DECREASE_BUDGET", donor.get("daily_budget", 0), new_budget):
                    continue
                plan_rows.append({
                    "portfolio_key": portfolio_key,
                    "portfolio_name": donor.get("portfolio_name", ""),
                    "campaign_id": donor.get("campaign_id", ""),
                    "campaign_name": donor.get("campaign_name", ""),
                    "daily_budget": donor.get("daily_budget", 0),
                    "recommended_daily_budget": new_budget,
                    "reallocation_delta": round(-donor_take, 2),
                    "campaign_action": "DECREASE_BUDGET",
                    "reallocation_role": "donor",
                    "confidence": donor.get("confidence", ""),
                    "score": donor.get("score", 0),
                    "reason": f"{donor.get('reason', '')} | portfolio_reallocation=out",
                })

            for receiver, receiver_give in receiver_rows:
                new_budget = round_budget_value(float(receiver.get("daily_budget", 0)) + receiver_give, self.max_budget_cap)
                if not self.action_clears_execution_threshold("INCREASE_BUDGET", receiver.get("daily_budget", 0), new_budget):
                    continue
                plan_rows.append({
                    "portfolio_key": portfolio_key,
                    "portfolio_name": receiver.get("portfolio_name", ""),
                    "campaign_id": receiver.get("campaign_id", ""),
                    "campaign_name": receiver.get("campaign_name", ""),
                    "daily_budget": receiver.get("daily_budget", 0),
                    "recommended_daily_budget": new_budget,
                    "reallocation_delta": round(receiver_give, 2),
                    "campaign_action": "INCREASE_BUDGET",
                    "reallocation_role": "receiver",
                    "confidence": receiver.get("confidence", ""),
                    "score": receiver.get("score", 0),
                    "reason": f"{receiver.get('reason', '')} | portfolio_reallocation=in",
                })

        return pd.DataFrame(plan_rows)


    def generate_placement_bulk_updates(self, targeting_with_share_df, bulk_campaigns_df, adjusted_min_roas):
        if targeting_with_share_df is None or targeting_with_share_df.empty:
            return pd.DataFrame()
        if "placement_bucket" not in targeting_with_share_df.columns:
            return pd.DataFrame()

        perf = targeting_with_share_df.copy()
        perf = perf[perf["placement_bucket"].astype(str).isin(["top_of_search", "product_pages", "rest_of_search"])].copy()
        if perf.empty:
            return pd.DataFrame()

        perf["campaign_intent"] = perf.apply(
            lambda r: self.classify_campaign_intent(r.get("campaign_name", ""), r.get("brand_segment", ""), r.get("funnel_stage", "")),
            axis=1,
        )

        placement_perf = (
            perf.groupby(["campaign_name", "placement_bucket", "campaign_intent"], as_index=False)
            .agg(clicks=("clicks", "sum"), orders=("orders", "sum"), spend=("spend", "sum"), sales=("sales", "sum"))
        )
        placement_perf["roas"] = np.where(placement_perf["spend"] > 0, placement_perf["sales"] / placement_perf["spend"], 0)

        campaigns = bulk_campaigns_df.copy()
        if campaigns.empty:
            return pd.DataFrame()

        recs = placement_perf.merge(campaigns, on="campaign_name", how="left")
        recs = recs[recs["campaign_id"].astype(str).str.strip() != ""].copy()
        if recs.empty:
            return pd.DataFrame()

        actions, pcts, reasons = [], [], []
        for _, row in recs.iterrows():
            roas = float(row.get("roas", 0) or 0)
            orders = float(row.get("orders", 0) or 0)
            clicks = float(row.get("clicks", 0) or 0)
            placement = str(row.get("placement_bucket", "") or "")
            campaign_intent = str(row.get("campaign_intent", "") or "")
            action = "NO_ACTION"
            pct = 0
            effective_target = self.get_effective_target(row, adjusted_min_roas)

            if placement == "top_of_search" and roas >= effective_target * 1.10 and orders >= self.min_orders_for_scaling:
                pct = 25 if campaign_intent in {"rank", "scale"} else 20
                action = "SET_PLACEMENT_MULTIPLIER"
            elif placement == "product_pages" and clicks >= self.min_clicks and roas < effective_target:
                pct = 0
                action = "SET_PLACEMENT_MULTIPLIER"
            elif placement == "rest_of_search" and roas >= effective_target and orders >= self.min_orders_for_scaling and campaign_intent in {"discovery", "scale", "balanced"}:
                pct = 10
                action = "SET_PLACEMENT_MULTIPLIER"

            pct = round_placement_pct(pct)
            if action != "NO_ACTION" and not self.action_clears_execution_threshold(action, 0, pct):
                action = "NO_ACTION"
                pct = 0

            actions.append(action)
            pcts.append(pct)
            reasons.append(f"placement={placement} | roas={round(roas,2)} | target={round(effective_target,2)} | intent={campaign_intent}")

        recs["placement_action"] = actions
        recs["placement_pct"] = pcts
        recs["reason"] = reasons
        recs = recs[recs["placement_action"] != "NO_ACTION"].copy()
        if recs.empty:
            return pd.DataFrame()

        bulk = pd.DataFrame(index=recs.index)
        bulk["Product"] = "Sponsored Products"
        bulk["Entity"] = "Campaign"
        bulk["Operation"] = "Update"
        bulk["Campaign ID"] = recs["campaign_id"]
        bulk["Ad Group ID"] = ""
        bulk["Keyword ID"] = ""
        bulk["Campaign Name"] = recs["campaign_name"]
        bulk["Ad Group Name"] = ""
        bulk["State"] = "Enabled"
        bulk["Keyword Text"] = ""
        bulk["Match Type"] = ""
        bulk["Bid"] = ""
        bulk["Daily Budget"] = ""
        bulk["Placement Type"] = recs["placement_bucket"]
        bulk["Placement %"] = recs["placement_pct"]
        bulk["Optimizer Action"] = recs["placement_action"]
        bulk["Reason"] = recs["reason"]
        return bulk.reset_index(drop=True)


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
        bulk["Confidence"] = actionable.get("confidence", "")
        bulk["Reason"] = actionable.get("reason", "")
        bulk["Score"] = actionable.get("score", "")

        return bulk.reset_index(drop=True)

    def generate_harvest_bulk_updates(self, search_term_actions_df):
        if not hasattr(self, "harvested_exact_memory"):
            self.harvested_exact_memory = set()

        actionable = search_term_actions_df[
            search_term_actions_df["search_term_action"] == "HARVEST_TO_EXACT"
        ].copy()

        if actionable.empty:
            return pd.DataFrame()

        actionable["campaign_id_norm"] = actionable["campaign_id"].map(normalize_entity_id)
        actionable["ad_group_id_norm"] = actionable["ad_group_id"].map(normalize_entity_id)

        actionable = actionable[actionable["campaign_id_norm"] != ""]
        actionable = actionable[actionable["ad_group_id_norm"] != ""]

        actionable["normalized_term"] = (
            actionable["search_term"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        actionable["canonical_term"] = actionable["search_term"].apply(canonicalize_term)

        actionable["campaign_name_norm"] = (
            actionable["campaign_name"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        actionable["ad_group_name_norm"] = (
            actionable["ad_group_name"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        actionable["ad_group_key"] = actionable["campaign_name_norm"] + "||" + actionable["ad_group_name_norm"]
        actionable = actionable[actionable["normalized_term"] != ""]
        actionable = actionable[actionable["ad_group_key"].isin(self.keyword_capable_ad_groups)]

        if actionable.empty:
            return pd.DataFrame()

        if "orders" in actionable.columns:
            actionable["orders"] = pd.to_numeric(actionable["orders"], errors="coerce").fillna(0)
            actionable = actionable[actionable["orders"] >= 2].copy()

        if actionable.empty:
            return pd.DataFrame()

        actionable["exact_dupe_key_id"] = (
            actionable["campaign_id_norm"]
            + "||"
            + actionable["ad_group_id_norm"]
            + "||"
            + actionable["canonical_term"]
            + "||exact"
        )
        actionable["exact_dupe_key_name"] = (
            actionable["campaign_name_norm"]
            + "||"
            + actionable["ad_group_name_norm"]
            + "||"
            + actionable["canonical_term"]
            + "||exact"
        )
        actionable["memory_key"] = (
            actionable["campaign_name_norm"]
            + "||"
            + actionable["ad_group_name_norm"]
            + "||"
            + actionable["canonical_term"]
        )

        existing_exact_keys_id = set()
        existing_exact_keys_name = set()
        existing_exact_terms_by_group = {}

        if hasattr(self, "bulk_df") and self.bulk_df is not None and not self.bulk_df.empty:
            bulk = self.bulk_df.copy()
            bulk.columns = [str(c).strip() for c in bulk.columns]

            if "Entity" in bulk.columns:
                exact_rows = bulk[bulk["Entity"].astype(str).str.strip().eq("Keyword")].copy()

                if not exact_rows.empty:
                    for required_col in ["Campaign ID", "Ad Group ID", "Keyword Text", "Match Type", "Campaign Name", "Ad Group Name"]:
                        if required_col not in exact_rows.columns:
                            exact_rows[required_col] = ""

                    exact_rows["campaign_id_norm"] = exact_rows["Campaign ID"].map(normalize_entity_id)
                    exact_rows["ad_group_id_norm"] = exact_rows["Ad Group ID"].map(normalize_entity_id)
                    exact_rows["campaign_name_norm"] = exact_rows["Campaign Name"].fillna("").astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
                    exact_rows["ad_group_name_norm"] = exact_rows["Ad Group Name"].fillna("").astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
                    exact_rows["normalized_term"] = (
                        exact_rows["Keyword Text"]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .str.strip()
                        .str.replace(r"\s+", " ", regex=True)
                    )
                    exact_rows["canonical_term"] = exact_rows["Keyword Text"].apply(canonicalize_term)
                    exact_rows["normalized_match"] = (
                        exact_rows["Match Type"]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )

                    exact_rows = exact_rows[exact_rows["normalized_match"] == "exact"].copy()

                    existing_exact_keys_id = set(
                        exact_rows["campaign_id_norm"]
                        + "||"
                        + exact_rows["ad_group_id_norm"]
                        + "||"
                        + exact_rows["canonical_term"]
                        + "||exact"
                    )
                    existing_exact_keys_name = set(
                        exact_rows["campaign_name_norm"]
                        + "||"
                        + exact_rows["ad_group_name_norm"]
                        + "||"
                        + exact_rows["canonical_term"]
                        + "||exact"
                    )

                    for _, r in exact_rows.iterrows():
                        group_key = r["campaign_name_norm"] + "||" + r["ad_group_name_norm"]
                        existing_exact_terms_by_group.setdefault(group_key, set()).add(r["canonical_term"])

        actionable = actionable[
            ~actionable["exact_dupe_key_id"].isin(existing_exact_keys_id)
            & ~actionable["exact_dupe_key_name"].isin(existing_exact_keys_name)
        ].copy()

        if hasattr(self, "existing_any_keywords") and self.existing_any_keywords:
            actionable["legacy_name_key"] = (
                actionable["campaign_name_norm"] + "||" + actionable["ad_group_name_norm"] + "||" + actionable["normalized_term"]
            )
            actionable = actionable[~actionable["legacy_name_key"].isin(self.existing_any_keywords)].copy()

        actionable = actionable[~actionable["memory_key"].isin(self.harvested_exact_memory)].copy()

        if actionable.empty:
            return pd.DataFrame()

        # semantic duplicate suppression within each campaign/ad group
        keep_indices = []
        for idx, row in actionable.iterrows():
            group_key = row["campaign_name_norm"] + "||" + row["ad_group_name_norm"]
            candidate_term = row["canonical_term"]

            existing_terms = existing_exact_terms_by_group.get(group_key, set())
            if any(is_semantic_duplicate(candidate_term, existing_term) for existing_term in existing_terms):
                continue

            if any(is_semantic_duplicate(candidate_term, actionable.loc[k, "canonical_term"]) for k in keep_indices):
                continue

            keep_indices.append(idx)

        actionable = actionable.loc[keep_indices].copy()

        actionable = actionable.drop_duplicates(
            subset=["campaign_id_norm", "ad_group_id_norm", "canonical_term"],
            keep="first",
        )

        if actionable.empty:
            return pd.DataFrame()

        if not hasattr(self, "harvested_exact_memory"):
            self.harvested_exact_memory = set()
        self.harvested_exact_memory.update(actionable["memory_key"].tolist())

        bulk = pd.DataFrame(index=actionable.index)
        bulk["Product"] = "Sponsored Products"
        bulk["Entity"] = "Keyword"
        bulk["Operation"] = "Create"
        bulk["Campaign ID"] = actionable["campaign_id_norm"]
        bulk["Ad Group ID"] = actionable["ad_group_id_norm"]
        bulk["Keyword ID"] = ""
        bulk["Campaign Name"] = actionable["campaign_name"]
        bulk["Ad Group Name"] = actionable["ad_group_name"]
        bulk["State"] = "Enabled"
        bulk["Keyword Text"] = actionable["normalized_term"]
        bulk["Match Type"] = "Exact"
        bulk["Bid"] = actionable["recommended_bid"]
        bulk["Daily Budget"] = ""
        bulk["Optimizer Action"] = actionable["search_term_action"]
        bulk["Confidence"] = actionable.get("confidence", "")
        bulk["Reason"] = actionable.get("reason", "")
        bulk["Score"] = ""

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
        bulk["Confidence"] = actionable.get("confidence", "")
        bulk["Reason"] = actionable.get("reason", "")
        bulk["Score"] = ""

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
        bulk["Confidence"] = actionable.get("confidence", "")
        bulk["Reason"] = actionable.get("reason", "")
        bulk["Score"] = ""

        return bulk.reset_index(drop=True)

    # -----------------------------
    # SAFEGUARDS
    # -----------------------------
    def apply_final_safeguards(self, combined_bulk_updates):
        df = _dedupe_and_strip_columns(combined_bulk_updates)

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
            "Confidence",
            "Reason",
            "Entity",
            "Operation",
            "State",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
                df[col] = df[col].str.replace(r"\.0$", "", regex=True)

        if "Optimizer Action" in df.columns:
            df = df[df["Optimizer Action"].fillna("").astype(str).str.upper() != "NO_ACTION"].copy()

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
            "Bid",
            "Daily Budget",
        ]
        existing_signature_cols = [c for c in signature_cols if c in df.columns]

        if existing_signature_cols:
            df = df.drop_duplicates(subset=existing_signature_cols, keep="last")

        df = _dedupe_and_strip_columns(df)
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
        high_conf = int((combined_bulk_updates.get("Confidence", pd.Series(dtype=str)) == "HIGH").sum()) if isinstance(combined_bulk_updates, pd.DataFrame) else 0
        low_conf = int((combined_bulk_updates.get("Confidence", pd.Series(dtype=str)) == "LOW").sum()) if isinstance(combined_bulk_updates, pd.DataFrame) else 0

        estimated_spend_impact_pct = round(
            (bid_increases * self.max_bid_up * 100)
            - (bid_decreases * self.max_bid_down * 100)
            + (budget_increases * self.budget_up_pct * 100)
            - (budget_decreases * self.budget_down_pct * 100),
            2,
        )

        simulation_mode = "Balanced Simulation"
        if estimated_spend_impact_pct >= 10:
            simulation_mode = "Growth Simulation"
        elif estimated_spend_impact_pct <= -10:
            simulation_mode = "Efficiency Simulation"

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
            "high_confidence_actions": high_conf,
            "low_confidence_actions": low_conf,
            "simulation_mode": simulation_mode,
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
        sqp = self.normalize_sqp()

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
        bid_recommendations = ensure_trend_columns(ensure_score_column(bid_recommendations))

        search_term_actions = self.build_search_term_actions(
            search_terms,
            adjusted_min_roas,
        )
        search_term_actions = ensure_trend_columns(ensure_score_column(search_term_actions))

        campaign_budget_actions = self.build_campaign_budget_actions(
            targeting_with_share,
            bulk_campaigns,
            adjusted_min_roas,
        )
        campaign_budget_actions = ensure_trend_columns(ensure_score_column(campaign_budget_actions))

        portfolio_budget_reallocation_plan = self.build_portfolio_budget_reallocation_plan(campaign_budget_actions)
        if not portfolio_budget_reallocation_plan.empty:
            campaign_budget_actions = portfolio_budget_reallocation_plan.copy()

        bid_bulk_updates = self.generate_bid_bulk_updates(bid_recommendations)
        harvest_bulk_updates = self.generate_harvest_bulk_updates(search_term_actions)
        negative_bulk_updates = self.generate_negative_bulk_updates(search_term_actions)
        budget_bulk_updates = self.generate_budget_bulk_updates(campaign_budget_actions)
        placement_bulk_updates = self.generate_placement_bulk_updates(targeting_with_share, bulk_campaigns, adjusted_min_roas)

        combined_bulk_updates = safe_concat_frames(
            [
                bid_bulk_updates,
                harvest_bulk_updates,
                negative_bulk_updates,
                budget_bulk_updates,
                placement_bulk_updates,
            ],
            ignore_index=True,
        )

        combined_bulk_updates = self.apply_final_safeguards(combined_bulk_updates)
        simulation_summary = self.build_simulation_summary(combined_bulk_updates, account_health)

        campaign_health_dashboard = self.build_campaign_health_dashboard(
            targeting_with_share,
            adjusted_min_roas,
        )
        campaign_health_dashboard = ensure_trend_columns(ensure_score_column(campaign_health_dashboard))

        sqp_opportunities, sqp_summary = self.build_sqp_opportunities(
            sqp_df=sqp,
            search_terms_df=search_terms,
        )

        smart = self.build_smart_warnings(
            targeting_with_share_df=targeting_with_share,
            search_terms_df=search_terms,
            campaign_health_df=campaign_health_dashboard,
            account_health=account_health,
            adjusted_min_roas=adjusted_min_roas,
            sqp_summary=sqp_summary,
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

        bid_recommendations = ensure_trend_columns(ensure_score_column(bid_recommendations))

        top_opportunities = bid_recommendations[bid_recommendations["recommended_action"] == "INCREASE_BID"].copy()
        if not top_opportunities.empty:
            top_opportunities = ensure_trend_columns(ensure_score_column(top_opportunities))
            top_opportunities = top_opportunities.sort_values(by=["score", "roas", "orders"], ascending=[False, False, False]).head(25)

        self.save_run_history(simulation_summary, account_health)
        self.save_action_history(bid_recommendations, entity_level="target")
        self.save_action_history(search_term_actions, entity_level="target")
        self.save_action_history(campaign_budget_actions, entity_level="campaign")

        bid_recommendations = ensure_trend_columns(ensure_score_column(bid_recommendations))
        search_term_actions = ensure_trend_columns(ensure_score_column(search_term_actions))
        campaign_budget_actions = ensure_trend_columns(ensure_score_column(campaign_budget_actions))
        top_opportunities = ensure_trend_columns(ensure_score_column(top_opportunities)) if 'top_opportunities' in locals() else pd.DataFrame()

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
            "top_opportunities": top_opportunities,
            "budget_reallocation_plan": portfolio_budget_reallocation_plan if 'portfolio_budget_reallocation_plan' in locals() else pd.DataFrame(),
        }

# =========================================================
# Phase 2 additive multi-ad-type orchestration
# =========================================================

class Phase2UploadValidator(Phase1UploadValidator):
    """Phase 2 validator currently reuses the Phase 1 validation logic."""
    pass

class _Phase2BaseOptimizer:
    def __init__(
        self,
        bulk_file,
        min_roas=3.0,
        min_clicks=8,
        zero_order_click_threshold=12,
        zero_order_action="Both",
        strategy_mode="Balanced",
        enable_bid_updates=True,
        enable_search_harvesting=True,
        enable_negative_keywords=True,
        enable_budget_updates=True,
        max_bid_cap=5.0,
        max_budget_cap=500.0,
        tacos_constrained=False,
    ):
        self.bulk_file = bulk_file
        self.min_roas = float(min_roas)
        self.min_clicks = int(min_clicks)
        self.zero_order_click_threshold = int(zero_order_click_threshold)
        self.zero_order_action = zero_order_action
        self.strategy_mode = strategy_mode
        self.enable_bid_updates = enable_bid_updates
        self.enable_search_harvesting = enable_search_harvesting
        self.enable_negative_keywords = enable_negative_keywords
        self.enable_budget_updates = enable_budget_updates
        self.max_bid_cap = float(max_bid_cap)
        self.max_budget_cap = float(max_budget_cap)
        self.tacos_constrained = bool(tacos_constrained)
        self.apply_strategy_settings()

    def apply_strategy_settings(self):
        mode = str(self.strategy_mode).strip().lower()
        if mode == 'conservative':
            self.max_bid_up = 0.05
            self.max_bid_down = 0.10
            self.scale_roas_multiplier = 1.40
            self.budget_up_pct = 0.05
            self.budget_down_pct = 0.08
            self.min_orders_for_scaling = 3
        elif mode == 'aggressive':
            self.max_bid_up = 0.20
            self.max_bid_down = 0.25
            self.scale_roas_multiplier = 1.10
            self.budget_up_pct = 0.15
            self.budget_down_pct = 0.12
            self.min_orders_for_scaling = 2
        else:
            self.max_bid_up = 0.10
            self.max_bid_down = 0.15
            self.scale_roas_multiplier = 1.25
            self.budget_up_pct = 0.10
            self.budget_down_pct = 0.10
            self.min_orders_for_scaling = 2

    def _clone_file_obj(self, file_obj):
        if file_obj is None:
            return None
        if isinstance(file_obj, (str, bytes, os.PathLike)):
            return file_obj
        if isinstance(file_obj, io.BytesIO):
            return io.BytesIO(file_obj.getvalue())
        if hasattr(file_obj, 'getvalue'):
            return io.BytesIO(file_obj.getvalue())
        if hasattr(file_obj, 'read'):
            try:
                pos = file_obj.tell()
            except Exception:
                pos = None
            try:
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                data = file_obj.read()
            finally:
                try:
                    if pos is not None and hasattr(file_obj, 'seek'):
                        file_obj.seek(pos)
                except Exception:
                    pass
            return io.BytesIO(data)
        raise ValueError('Unsupported file object type.')

    def load_file(self, file_obj):
        if file_obj is None:
            return None
        name = getattr(file_obj, 'name', '') if not isinstance(file_obj, (str, os.PathLike)) else str(file_obj)
        ext = os.path.splitext(name)[1].lower()
        cloned = self._clone_file_obj(file_obj)
        if ext == '.csv':
            return pd.read_csv(cloned)
        return pd.read_excel(cloned, engine='openpyxl')

    def load_bulk_workbook(self):
        return pd.read_excel(self._clone_file_obj(self.bulk_file), sheet_name=None, engine='openpyxl')

    def safe_numeric(self, series):
        if series is None:
            return pd.Series(dtype=float)
        cleaned = (
            series.fillna('')
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('$', '', regex=False)
            .str.replace('%', '', regex=False)
            .str.replace(r'\(([^\)]+)\)', r'-\1', regex=True)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors='coerce').fillna(0)

    def first_present(self, df, cols, default=0.0):
        for col in cols:
            if col in df.columns:
                return self.safe_numeric(df[col])
        return pd.Series(default, index=df.index if hasattr(df, 'index') else None, dtype=float)

    def clean_text(self, series):
        return series.fillna('').astype(str).str.strip()

    def normalize_join_text(self, series):
        return (
            series.fillna('')
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'\s+', ' ', regex=True)
        )

    def _budget_action(self, roas, orders):
        if (
            not self.tacos_constrained
            and orders >= self.min_orders_for_scaling
            and roas >= self.min_roas * self.scale_roas_multiplier
        ):
            return 'INCREASE_BUDGET'
        if roas > 0 and roas < self.min_roas:
            return 'DECREASE_BUDGET'
        return None

    def _bid_action(self, roas, clicks, orders):
        if clicks >= self.zero_order_click_threshold and orders == 0:
            if self.zero_order_action in {'Decrease Bid', 'Both'}:
                return 'DECREASE_BID'
        if clicks >= self.min_clicks and roas > 0 and roas < self.min_roas:
            return 'DECREASE_BID'
        if (
            not self.tacos_constrained
            and orders >= self.min_orders_for_scaling
            and roas >= self.min_roas * self.scale_roas_multiplier
        ):
            return 'INCREASE_BID'
        return None

    def _adjust_bid(self, current_bid, action):
        bid = float(current_bid or 0)
        if bid <= 0:
            return 0.0
        if action == 'INCREASE_BID':
            bid = round_bid_value(bid * (1 + self.max_bid_up), self.max_bid_cap)
        elif action == 'DECREASE_BID':
            bid = round_bid_value(bid * (1 - self.max_bid_down), self.max_bid_cap)
        return round(bid, 2)

    def _adjust_budget(self, current_budget, action):
        budget = float(current_budget or 0)
        if budget <= 0:
            return 0.0
        if action == 'INCREASE_BUDGET':
            budget = round_budget_value(budget * (1 + self.budget_up_pct), self.max_budget_cap)
        elif action == 'DECREASE_BUDGET':
            budget = round_budget_value(budget * (1 - self.budget_down_pct), self.max_budget_cap)
        return round(budget, 2)

class SponsoredBrandsOptimizer(_Phase2BaseOptimizer):
    def __init__(self, bulk_file, search_term_file=None, impression_share_file=None, **kwargs):
        super().__init__(bulk_file=bulk_file, **kwargs)
        self.search_term_file = search_term_file
        self.impression_share_file = impression_share_file

    def _normalize_term(self, value):
        return " ".join(str(value or "").strip().lower().split())

    def _clean_scalar(self, value):
        if pd.isna(value):
            return ""
        return str(value).strip().replace(".0", "")

    def _key_by_id(self, campaign_id, ad_group_id, term, match_type):
        return "||".join([
            self._clean_scalar(campaign_id),
            self._clean_scalar(ad_group_id),
            self._normalize_term(term),
            self._normalize_term(match_type),
        ])

    def _key_by_name(self, campaign_name, ad_group_name, term, match_type):
        return "||".join([
            self._normalize_term(campaign_name),
            self._normalize_term(ad_group_name),
            self._normalize_term(term),
            self._normalize_term(match_type),
        ])

    def _ad_group_key(self, campaign_id, ad_group_id, campaign_name, ad_group_name):
        id_key = "||".join([self._clean_scalar(campaign_id), self._clean_scalar(ad_group_id)])
        name_key = "||".join([self._normalize_term(campaign_name), self._normalize_term(ad_group_name)])
        return id_key, name_key

    def _is_valid_harvest_term(self, value):
        term = self._normalize_term(value)
        if term == "":
            return False
        if len(term) < 3:
            return False
        if len(term) > 80:
            return False
        if len(term.split()) < 2:
            return False
        bad_chars = set(['[', ']', '{', '}', '|', ';'])
        if any(ch in term for ch in bad_chars):
            return False
        return True

    def _load_search_terms(self):
        df = self.load_file(self.search_term_file) if self.search_term_file is not None else None
        if df is None or df.empty:
            wb = self.load_bulk_workbook()
            df = wb.get('SB Search Term Report')

        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()

        out['campaign_id'] = self.clean_text(out.get('Campaign ID', pd.Series('', index=out.index)))
        out['ad_group_id'] = self.clean_text(out.get('Ad Group ID', pd.Series('', index=out.index)))
        out['keyword_id'] = self.clean_text(out.get('Keyword ID', pd.Series('', index=out.index)))
        out['product_targeting_id'] = self.clean_text(out.get('Product Targeting ID', pd.Series('', index=out.index)))

        out['campaign_name'] = self.clean_text(
            out.get('Campaign Name (Informational only)', out.get('Campaign Name', pd.Series('', index=out.index)))
        )

        out['ad_group_name'] = self.clean_text(
            out.get('Ad Group Name (Informational only)', out.get('Ad Group Name', pd.Series('', index=out.index)))
        )

        out['keyword_text'] = self.clean_text(
            out.get('Keyword Text', out.get('Keyword', out.get('Targeting', pd.Series('', index=out.index))))
        )

        out['match_type'] = self.clean_text(
            out.get('Match Type', out.get('Targeting Type', pd.Series('', index=out.index)))
        )

        out['search_term'] = self.clean_text(
            out.get('Customer Search Term', out.get('Search Term', pd.Series('', index=out.index)))
        )

        out['bid'] = self.first_present(out, ['Bid', 'Default Bid'])
        out['clicks'] = self.first_present(out, ['Clicks'])
        out['spend'] = self.first_present(out, ['Spend', 'Cost'])
        out['sales'] = self.first_present(out, ['Sales', 'Attributed Sales 14 Day', 'Attributed Sales'])
        out['orders'] = self.first_present(out, ['Orders', 'Purchases', 'Attributed Conversions 14 Day'])
        out['roas'] = self.first_present(out, ['ROAS', 'Return on Ad Spend'])
        out['cpc'] = self.first_present(out, ['CPC', 'Cost Per Click'])

        if 'ROAS' not in out.columns and 'Return on Ad Spend' not in out.columns:
            out['roas'] = np.where(out['spend'] > 0, out['sales'] / out['spend'], 0)

        out['campaign_name_key'] = self.normalize_join_text(out['campaign_name'])
        out['ad_group_name_key'] = self.normalize_join_text(out['ad_group_name'])
        out['keyword_text_key'] = self.normalize_join_text(out['keyword_text'])
        out['search_term_key'] = self.normalize_join_text(out['search_term'])

        return out

    def _bulk_sheets(self):
        wb = self.load_bulk_workbook()
        standard_name = find_matching_sheet_name(
            wb.keys(),
            ['Sponsored Brands Campaigns', 'Sponsored Brands', 'SB Campaigns', 'SB']
        )
        multi_name = find_matching_sheet_name(
            wb.keys(),
            ['SB Multi Ad Group Campaigns', 'Sponsored Brands Multi Ad Group Campaigns']
        )
        standard_sheet = wb.get(standard_name) if standard_name else None
        multi_sheet = wb.get(multi_name) if multi_name else None
        return standard_sheet, multi_sheet, wb

    def _build_existing_state(self, *sheet_candidates):
        existing_exact_by_id = set()
        existing_exact_by_name = set()
        existing_negative_by_id = set()
        existing_negative_by_name = set()
        keyword_capable_ad_groups = set()

        for sheet_df in sheet_candidates:
            if sheet_df is None or sheet_df.empty or 'Entity' not in sheet_df.columns:
                continue

            work = sheet_df.copy()
            campaign_names = self.clean_text(
                work.get('Campaign Name', work.get('Campaign Name (Informational only)', pd.Series('', index=work.index)))
            )
            ad_group_names = self.clean_text(
                work.get('Ad Group Name', work.get('Ad Group Name (Informational only)', pd.Series('', index=work.index)))
            )
            campaign_ids = self.clean_text(work.get('Campaign ID', pd.Series('', index=work.index)))
            ad_group_ids = self.clean_text(work.get('Ad Group ID', pd.Series('', index=work.index)))
            keyword_texts = self.clean_text(work.get('Keyword Text', pd.Series('', index=work.index)))
            match_types = self.clean_text(work.get('Match Type', pd.Series('', index=work.index)))

            entities = work['Entity'].fillna('').astype(str)

            keyword_rows = work[entities.str.fullmatch('Keyword', case=False)].copy()
            if not keyword_rows.empty:
                for idx in keyword_rows.index:
                    existing_exact_by_id.add(
                        self._key_by_id(campaign_ids.loc[idx], ad_group_ids.loc[idx], keyword_texts.loc[idx], match_types.loc[idx])
                    )
                    existing_exact_by_name.add(
                        self._key_by_name(campaign_names.loc[idx], ad_group_names.loc[idx], keyword_texts.loc[idx], match_types.loc[idx])
                    )
                    id_group_key, name_group_key = self._ad_group_key(
                        campaign_ids.loc[idx], ad_group_ids.loc[idx], campaign_names.loc[idx], ad_group_names.loc[idx]
                    )
                    keyword_capable_ad_groups.add(id_group_key)
                    keyword_capable_ad_groups.add(name_group_key)

            negative_rows = work[entities.str.fullmatch('Negative Keyword', case=False)].copy()
            if not negative_rows.empty:
                for idx in negative_rows.index:
                    existing_negative_by_id.add(
                        self._key_by_id(campaign_ids.loc[idx], ad_group_ids.loc[idx], keyword_texts.loc[idx], match_types.loc[idx])
                    )
                    existing_negative_by_name.add(
                        self._key_by_name(campaign_names.loc[idx], ad_group_names.loc[idx], keyword_texts.loc[idx], match_types.loc[idx])
                    )

            ad_group_rows = work[work.get('Ad Group ID', pd.Series('', index=work.index)).notna()].copy()
            if not ad_group_rows.empty:
                for idx in ad_group_rows.index:
                    id_group_key, name_group_key = self._ad_group_key(
                        campaign_ids.loc[idx], ad_group_ids.loc[idx], campaign_names.loc[idx], ad_group_names.loc[idx]
                    )
                    keyword_capable_ad_groups.add(id_group_key)
                    keyword_capable_ad_groups.add(name_group_key)

        return {
            'existing_exact_by_id': existing_exact_by_id,
            'existing_exact_by_name': existing_exact_by_name,
            'existing_negative_by_id': existing_negative_by_id,
            'existing_negative_by_name': existing_negative_by_name,
            'keyword_capable_ad_groups': keyword_capable_ad_groups,
        }

    def _attach_ids_from_bulk(self, perf_df, *sheet_candidates):
        if perf_df is None or perf_df.empty:
            return pd.DataFrame()

        lookup_frames = []
        for sheet_df in sheet_candidates:
            if sheet_df is None or sheet_df.empty:
                continue
            work = sheet_df.copy()
            campaign_name = self.clean_text(
                work.get('Campaign Name', work.get('Campaign Name (Informational only)', pd.Series('', index=work.index)))
            )
            ad_group_name = self.clean_text(
                work.get('Ad Group Name', work.get('Ad Group Name (Informational only)', pd.Series('', index=work.index)))
            )
            lookup = pd.DataFrame({
                'campaign_name_key': self.normalize_join_text(campaign_name),
                'ad_group_name_key': self.normalize_join_text(ad_group_name),
                'bulk_campaign_id': self.clean_text(work.get('Campaign ID', pd.Series('', index=work.index))),
                'bulk_ad_group_id': self.clean_text(work.get('Ad Group ID', pd.Series('', index=work.index))),
            })
            lookup = lookup[(lookup['campaign_name_key'] != '') & (lookup['ad_group_name_key'] != '')]
            if not lookup.empty:
                lookup_frames.append(lookup.drop_duplicates(subset=['campaign_name_key', 'ad_group_name_key']))

        if not lookup_frames:
            return perf_df.copy()

        lookup_df = safe_concat_frames(lookup_frames, ignore_index=True)
        if lookup_df.empty:
            return perf_df.copy()
        lookup_df = lookup_df.drop_duplicates(subset=['campaign_name_key', 'ad_group_name_key'], keep='first')

        out = perf_df.copy()
        out = out.merge(lookup_df, on=['campaign_name_key', 'ad_group_name_key'], how='left')
        out['campaign_id'] = np.where(
            out['campaign_id'].fillna('').astype(str).str.strip() != '',
            out['campaign_id'],
            out['bulk_campaign_id'].fillna('')
        )
        out['ad_group_id'] = np.where(
            out['ad_group_id'].fillna('').astype(str).str.strip() != '',
            out['ad_group_id'],
            out['bulk_ad_group_id'].fillna('')
        )
        return out.drop(columns=[c for c in ['bulk_campaign_id', 'bulk_ad_group_id'] if c in out.columns])

    def _backfill_action_ids(self, action_df, standard_sheet=None, multi_sheet=None):
        if action_df is None or action_df.empty:
            return pd.DataFrame()

        lookup_frames = []

        for sheet_df in [standard_sheet, multi_sheet]:
            if sheet_df is None or sheet_df.empty:
                continue

            work = sheet_df.copy()
            lookup = pd.DataFrame({
                'campaign_name_key': self.normalize_join_text(
                    self.clean_text(work.get('Campaign Name', work.get('Campaign Name (Informational only)', pd.Series('', index=work.index))))
                ),
                'ad_group_name_key': self.normalize_join_text(
                    self.clean_text(work.get('Ad Group Name', work.get('Ad Group Name (Informational only)', pd.Series('', index=work.index))))
                ),
                'bulk_campaign_id': self.clean_text(work.get('Campaign ID', pd.Series('', index=work.index))),
                'bulk_ad_group_id': self.clean_text(work.get('Ad Group ID', pd.Series('', index=work.index))),
            })
            lookup = lookup[
                (lookup['campaign_name_key'] != '')
                & (lookup['ad_group_name_key'] != '')
                & (lookup['bulk_campaign_id'] != '')
                & (lookup['bulk_ad_group_id'] != '')
            ]
            if not lookup.empty:
                lookup_frames.append(
                    lookup.drop_duplicates(subset=['campaign_name_key', 'ad_group_name_key'], keep='first')
                )

        if not lookup_frames:
            return action_df.copy()

        lookup_df = safe_concat_frames(lookup_frames, ignore_index=True)
        if lookup_df.empty:
            return action_df.copy()

        out = action_df.copy()
        out['campaign_name_key'] = self.normalize_join_text(out['campaign_name'])
        out['ad_group_name_key'] = self.normalize_join_text(out['ad_group_name'])
        out = out.merge(lookup_df, on=['campaign_name_key', 'ad_group_name_key'], how='left')

        out['campaign_id'] = np.where(
            out['campaign_id'].fillna('').astype(str).str.strip() != '',
            out['campaign_id'],
            out['bulk_campaign_id'].fillna('')
        )
        out['ad_group_id'] = np.where(
            out['ad_group_id'].fillna('').astype(str).str.strip() != '',
            out['ad_group_id'],
            out['bulk_ad_group_id'].fillna('')
        )

        return out.drop(columns=[
            c for c in ['campaign_name_key', 'ad_group_name_key', 'bulk_campaign_id', 'bulk_ad_group_id']
            if c in out.columns
        ])

    def _build_search_term_actions(self, perf_df, existing_state):
        if perf_df is None or perf_df.empty:
            return pd.DataFrame()

        df = perf_df.copy()
        actions = []
        recommended_bids = []

        for _, row in df.iterrows():
            action = 'NO_ACTION'
            search_term = self._normalize_term(row.get('search_term', ''))
            match_type = self._normalize_term(row.get('match_type', ''))
            cpc = float(row.get('cpc', 0) or 0)
            rec_bid = min(max(round(cpc * 1.10, 2), 0.20), self.max_bid_cap)

            id_group_key, name_group_key = self._ad_group_key(
                row.get('campaign_id', ''),
                row.get('ad_group_id', ''),
                row.get('campaign_name', ''),
                row.get('ad_group_name', ''),
            )

            exact_id_key = self._key_by_id(row.get('campaign_id', ''), row.get('ad_group_id', ''), search_term, 'exact')
            exact_name_key = self._key_by_name(row.get('campaign_name', ''), row.get('ad_group_name', ''), search_term, 'exact')
            negative_id_key = self._key_by_id(row.get('campaign_id', ''), row.get('ad_group_id', ''), search_term, 'negative phrase')
            negative_name_key = self._key_by_name(row.get('campaign_name', ''), row.get('ad_group_name', ''), search_term, 'negative phrase')

            clicks = float(row.get('clicks', 0) or 0)
            spend = float(row.get('spend', 0) or 0)
            orders = float(row.get('orders', 0) or 0)
            sales = float(row.get('sales', 0) or 0)
            roas = float(row.get('roas', 0) or 0)

            eligible_group = (
                id_group_key in existing_state['keyword_capable_ad_groups']
                or name_group_key in existing_state['keyword_capable_ad_groups']
            )

            if (
                self.enable_negative_keywords
                and self._is_valid_harvest_term(search_term)
                and ((clicks >= 4) or (spend >= 12))
                and orders == 0
                and sales == 0
                and negative_id_key not in existing_state['existing_negative_by_id']
                and negative_name_key not in existing_state['existing_negative_by_name']
            ):
                action = 'ADD_NEGATIVE_PHRASE'

            elif (
                self.enable_search_harvesting
                and eligible_group
                and self._is_valid_harvest_term(search_term)
                and search_term != ''
                and match_type != 'exact'
                and clicks >= 4
                and orders >= 1
                and roas >= max(self.min_roas, 1.5)
                and exact_id_key not in existing_state['existing_exact_by_id']
                and exact_name_key not in existing_state['existing_exact_by_name']
            ):
                action = 'HARVEST_TO_EXACT'

            actions.append(action)
            recommended_bids.append(rec_bid)

        df['search_term_action'] = actions
        df['recommended_bid'] = recommended_bids
        return df

    def _build_bid_updates_for_sheet(self, sheet_df, perf_df, id_col, entity_col='Entity'):
        if sheet_df is None or sheet_df.empty:
            return pd.DataFrame()

        if 'Bid' not in sheet_df.columns:
            return pd.DataFrame()

        work = sheet_df.copy()

        if entity_col in work.columns:
            work = work[
                work[entity_col].astype(str).str.contains('Keyword|Product Target', case=False, na=False)
            ].copy()

        if work.empty:
            return pd.DataFrame()

        work['_id_key'] = self.clean_text(work.get(id_col, pd.Series('', index=work.index)))
        work['_campaign_name_key'] = self.normalize_join_text(
            work.get('Campaign Name', work.get('Campaign Name (Informational only)', pd.Series('', index=work.index)))
        )
        work['_ad_group_name_key'] = self.normalize_join_text(
            work.get('Ad Group Name', work.get('Ad Group Name (Informational only)', pd.Series('', index=work.index)))
        )

        sheet_text_col = None
        for c in ['Keyword Text', 'Targeting', 'Product Targeting Expression', 'Resolved Expression', 'Resolved Product Targeting Expression (Informational only)']:
            if c in work.columns:
                sheet_text_col = c
                break

        if sheet_text_col is not None:
            work['_keyword_text_key'] = self.normalize_join_text(work[sheet_text_col])
        else:
            work['_keyword_text_key'] = ''

        perf = perf_df.copy()

        perf_id_col = {
            'Keyword ID': 'keyword_id',
            'Product Targeting ID': 'product_targeting_id',
        }.get(id_col, id_col)

        perf['_id_key'] = self.clean_text(perf.get(perf_id_col, pd.Series('', index=perf.index)))
        perf['_campaign_name_key'] = self.normalize_join_text(perf.get('campaign_name', pd.Series('', index=perf.index)))
        perf['_ad_group_name_key'] = self.normalize_join_text(perf.get('ad_group_name', pd.Series('', index=perf.index)))
        perf['_keyword_text_key'] = self.normalize_join_text(perf.get('keyword_text', pd.Series('', index=perf.index)))

        perf_by_id = perf[perf['_id_key'] != ''].groupby('_id_key', as_index=False).agg(
            clicks=('clicks', 'sum'),
            spend=('spend', 'sum'),
            sales=('sales', 'sum'),
            orders=('orders', 'sum'),
            roas=('roas', 'mean'),
        )

        joined = work.merge(perf_by_id, on='_id_key', how='left')

        fallback_perf = perf.groupby(
            ['_campaign_name_key', '_ad_group_name_key', '_keyword_text_key'],
            as_index=False
        ).agg(
            fb_clicks=('clicks', 'sum'),
            fb_spend=('spend', 'sum'),
            fb_sales=('sales', 'sum'),
            fb_orders=('orders', 'sum'),
            fb_roas=('roas', 'mean'),
        )

        joined = joined.merge(
            fallback_perf,
            on=['_campaign_name_key', '_ad_group_name_key', '_keyword_text_key'],
            how='left'
        )

        joined['clicks'] = joined['clicks'].fillna(joined['fb_clicks']).fillna(0)
        joined['spend'] = joined['spend'].fillna(joined['fb_spend']).fillna(0)
        joined['sales'] = joined['sales'].fillna(joined['fb_sales']).fillna(0)
        joined['orders'] = joined['orders'].fillna(joined['fb_orders']).fillna(0)
        joined['roas'] = joined['roas'].fillna(joined['fb_roas']).fillna(0)

        joined['current_bid'] = self.safe_numeric(joined['Bid'])

        def sb_bid_action(row):
            clicks = float(row['clicks'] or 0)
            orders = float(row['orders'] or 0)
            roas = float(row['roas'] or 0)
            spend = float(row['spend'] or 0)

            if (clicks >= 4 or spend >= 12) and orders == 0:
                return 'DECREASE_BID'
            if clicks >= 4 and roas > 0 and roas < self.min_roas:
                return 'DECREASE_BID'
            if orders >= self.min_orders_for_scaling and roas >= self.min_roas * 1.10:
                return 'INCREASE_BID'
            return None

        joined['optimizer_action'] = joined.apply(
            lambda r: sb_bid_action(r) if self.enable_bid_updates else None,
            axis=1,
        )

        joined = joined[joined['optimizer_action'].notna()].copy()
        if joined.empty:
            return pd.DataFrame()

        joined['new_bid'] = joined.apply(
            lambda r: self._adjust_bid(r['current_bid'], r['optimizer_action']),
            axis=1,
        )

        joined = joined[joined['new_bid'] != joined['current_bid']].copy()
        if joined.empty:
            return pd.DataFrame()

        joined['Bid'] = joined['new_bid']
        joined['Operation'] = 'update'
        joined['Product'] = 'Sponsored Brands'
        joined['ad_type'] = 'SB'
        joined['source_type'] = 'bid'
        joined['campaign'] = joined.get('Campaign Name', joined.get('campaign_name', ''))
        joined['ad_group'] = joined.get('Ad Group Name', joined.get('ad_group_name', ''))

        return joined

    def _build_ad_group_fallback_bid_updates(self, sheet_df, perf_df, entity_col='Entity'):
        if sheet_df is None or sheet_df.empty or perf_df is None or perf_df.empty:
            return pd.DataFrame()

        work = sheet_df.copy()

        if entity_col in work.columns:
            work = work[
                work[entity_col].astype(str).str.contains('Keyword|Product Target', case=False, na=False)
            ].copy()

        if work.empty or 'Bid' not in work.columns:
            return pd.DataFrame()

        work['_campaign_name_key'] = self.normalize_join_text(
            work.get('Campaign Name', work.get('Campaign Name (Informational only)', pd.Series('', index=work.index)))
        )
        work['_ad_group_name_key'] = self.normalize_join_text(
            work.get('Ad Group Name', work.get('Ad Group Name (Informational only)', pd.Series('', index=work.index)))
        )

        perf_grouped = perf_df.groupby(
            ['campaign_name_key', 'ad_group_name_key'],
            as_index=False
        ).agg(
            clicks=('clicks', 'sum'),
            spend=('spend', 'sum'),
            sales=('sales', 'sum'),
            orders=('orders', 'sum'),
            roas=('roas', 'mean'),
        )

        joined = work.merge(
            perf_grouped,
            left_on=['_campaign_name_key', '_ad_group_name_key'],
            right_on=['campaign_name_key', 'ad_group_name_key'],
            how='left'
        )

        joined['clicks'] = joined['clicks'].fillna(0)
        joined['spend'] = joined['spend'].fillna(0)
        joined['sales'] = joined['sales'].fillna(0)
        joined['orders'] = joined['orders'].fillna(0)
        joined['roas'] = joined['roas'].fillna(0)
        joined['current_bid'] = self.safe_numeric(joined['Bid'])

        def sb_bid_action(row):
            clicks = float(row['clicks'] or 0)
            orders = float(row['orders'] or 0)
            roas = float(row['roas'] or 0)
            spend = float(row['spend'] or 0)

            if (clicks >= 4 or spend >= 12) and orders == 0:
                return 'DECREASE_BID'
            if clicks >= 4 and roas > 0 and roas < self.min_roas:
                return 'DECREASE_BID'
            if orders >= self.min_orders_for_scaling and roas >= self.min_roas * 1.10:
                return 'INCREASE_BID'
            return None

        joined['optimizer_action'] = joined.apply(
            lambda r: sb_bid_action(r) if self.enable_bid_updates else None,
            axis=1,
        )

        joined = joined[joined['optimizer_action'].notna()].copy()
        if joined.empty:
            return pd.DataFrame()

        joined['new_bid'] = joined.apply(
            lambda r: self._adjust_bid(r['current_bid'], r['optimizer_action']),
            axis=1,
        )

        joined = joined[joined['new_bid'] != joined['current_bid']].copy()
        if joined.empty:
            return pd.DataFrame()

        joined['Bid'] = joined['new_bid']
        joined['Operation'] = 'update'
        joined['Product'] = 'Sponsored Brands'
        joined['ad_type'] = 'SB'
        joined['source_type'] = 'bid_fallback_ad_group'
        joined['campaign'] = joined.get('Campaign Name', '')
        joined['ad_group'] = joined.get('Ad Group Name', '')

        dedupe_cols = [c for c in ['Campaign ID', 'Ad Group ID', 'Keyword ID', 'Product Targeting ID', 'campaign', 'ad_group'] if c in joined.columns]
        if dedupe_cols:
            joined = joined.drop_duplicates(subset=dedupe_cols, keep='first')

        return joined

    def _build_campaign_budget_updates(self, sheet_df, perf_df, campaign_id_col='Campaign ID', budget_col='Budget'):
        if sheet_df is None or sheet_df.empty or budget_col not in sheet_df.columns:
            return pd.DataFrame()

        campaigns = sheet_df.copy()
        if 'Entity' in campaigns.columns:
            campaigns = campaigns[campaigns['Entity'].astype(str).str.contains('Campaign', case=False, na=False)].copy()
        if campaigns.empty:
            return pd.DataFrame()

        campaigns['_campaign_key'] = self.clean_text(campaigns.get(campaign_id_col, pd.Series('', index=campaigns.index)))
        campaigns['_campaign_name_key'] = self.normalize_join_text(
            campaigns.get('Campaign Name', campaigns.get('Campaign Name (Informational only)', pd.Series('', index=campaigns.index)))
        )

        perf_grouped = perf_df.groupby('campaign_id', as_index=False).agg(
            clicks=('clicks', 'sum'),
            spend=('spend', 'sum'),
            sales=('sales', 'sum'),
            orders=('orders', 'sum'),
            roas=('roas', 'mean'),
            campaign_name=('campaign_name', 'first'),
        )
        perf_grouped['_campaign_key'] = self.clean_text(perf_grouped['campaign_id'])

        joined = campaigns.merge(perf_grouped, on='_campaign_key', how='left')

        if {'clicks', 'spend', 'sales', 'orders', 'roas'}.issubset(joined.columns):
            perf_by_name = perf_df.groupby('campaign_name_key', as_index=False).agg(
                fb_clicks=('clicks', 'sum'),
                fb_spend=('spend', 'sum'),
                fb_sales=('sales', 'sum'),
                fb_orders=('orders', 'sum'),
                fb_roas=('roas', 'mean'),
            )
            joined = joined.merge(
                perf_by_name,
                left_on='_campaign_name_key',
                right_on='campaign_name_key',
                how='left'
            )
            joined['clicks'] = joined['clicks'].fillna(joined['fb_clicks']).fillna(0)
            joined['spend'] = joined['spend'].fillna(joined['fb_spend']).fillna(0)
            joined['sales'] = joined['sales'].fillna(joined['fb_sales']).fillna(0)
            joined['orders'] = joined['orders'].fillna(joined['fb_orders']).fillna(0)
            joined['roas'] = joined['roas'].fillna(joined['fb_roas']).fillna(0)
        else:
            joined['clicks'] = 0
            joined['spend'] = 0
            joined['sales'] = 0
            joined['orders'] = 0
            joined['roas'] = 0

        joined['current_budget'] = self.safe_numeric(joined[budget_col])

        def sb_budget_action(row):
            clicks = float(row['clicks'] or 0)
            orders = float(row['orders'] or 0)
            roas = float(row['roas'] or 0)
            spend = float(row['spend'] or 0)

            if (clicks >= 6 or spend >= 20) and orders == 0:
                return 'DECREASE_BUDGET'
            if roas > 0 and roas < self.min_roas:
                return 'DECREASE_BUDGET'
            if orders >= self.min_orders_for_scaling and roas >= self.min_roas * 1.10:
                return 'INCREASE_BUDGET'
            return None

        joined['optimizer_action'] = joined.apply(
            lambda r: sb_budget_action(r) if self.enable_budget_updates else None,
            axis=1,
        )
        joined = joined[joined['optimizer_action'].notna()].copy()
        if joined.empty:
            return pd.DataFrame()

        joined['new_budget'] = joined.apply(lambda r: self._adjust_budget(r['current_budget'], r['optimizer_action']), axis=1)
        joined = joined[joined['new_budget'] != joined['current_budget']].copy()
        if joined.empty:
            return pd.DataFrame()

        joined[budget_col] = joined['new_budget']
        joined['Operation'] = 'update'
        joined['Product'] = 'Sponsored Brands'
        joined['ad_type'] = 'SB'
        joined['source_type'] = 'budget'
        joined['campaign'] = joined.get('Campaign Name', joined.get('campaign_name', ''))
        joined['ad_group'] = ''
        return joined

    def _generate_harvest_bulk_updates(self, search_term_actions_df, standard_sheet=None, multi_sheet=None):
        actionable = search_term_actions_df[
            search_term_actions_df['search_term_action'] == 'HARVEST_TO_EXACT'
        ].copy()

        if actionable.empty:
            return pd.DataFrame()

        actionable = self._backfill_action_ids(
            actionable,
            standard_sheet=standard_sheet,
            multi_sheet=multi_sheet,
        )

        actionable['normalized_term'] = actionable['search_term'].apply(self._normalize_term)
        actionable = actionable[actionable['normalized_term'].apply(self._is_valid_harvest_term)]
        actionable = actionable[
            (actionable['campaign_name'].fillna('').astype(str).str.strip() != '')
            & (actionable['ad_group_name'].fillna('').astype(str).str.strip() != '')
        ]
        if actionable.empty:
            return pd.DataFrame()

        actionable['campaign_id'] = actionable['campaign_id'].fillna('').astype(str).str.strip()
        actionable['ad_group_id'] = actionable['ad_group_id'].fillna('').astype(str).str.strip()
        actionable = actionable[
            (actionable['campaign_id'] != '')
            & (actionable['ad_group_id'] != '')
        ]
        if actionable.empty:
            return pd.DataFrame()

        actionable = actionable.drop_duplicates(
            subset=['campaign_name', 'ad_group_name', 'normalized_term'],
            keep='first'
        )

        bulk = pd.DataFrame(index=actionable.index)
        bulk['Product'] = 'Sponsored Brands'
        bulk['Entity'] = 'Keyword'
        bulk['Operation'] = 'Create'
        bulk['Campaign ID'] = actionable['campaign_id']
        bulk['Ad Group ID'] = actionable['ad_group_id']
        bulk['Keyword ID'] = ''
        bulk['Product Targeting ID'] = ''
        bulk['Campaign Name'] = actionable['campaign_name']
        bulk['Ad Group Name'] = actionable['ad_group_name']
        bulk['State'] = 'enabled'
        bulk['Keyword Text'] = actionable['normalized_term']
        bulk['Match Type'] = 'Exact'
        bulk['Bid'] = actionable['recommended_bid']
        bulk['Budget'] = ''
        bulk['Optimizer Action'] = actionable['search_term_action']
        bulk['ad_type'] = 'SB'
        bulk['source_type'] = 'harvest'
        bulk['campaign'] = actionable['campaign_name']
        bulk['ad_group'] = actionable['ad_group_name']
        return bulk.reset_index(drop=True)

    def _generate_negative_bulk_updates(self, search_term_actions_df, standard_sheet=None, multi_sheet=None):
        actionable = search_term_actions_df[
            search_term_actions_df['search_term_action'] == 'ADD_NEGATIVE_PHRASE'
        ].copy()

        if actionable.empty:
            return pd.DataFrame()

        actionable = self._backfill_action_ids(
            actionable,
            standard_sheet=standard_sheet,
            multi_sheet=multi_sheet,
        )

        actionable['normalized_term'] = actionable['search_term'].apply(self._normalize_term)
        actionable = actionable[actionable['normalized_term'].apply(self._is_valid_harvest_term)]
        actionable = actionable[
            (actionable['campaign_name'].fillna('').astype(str).str.strip() != '')
            & (actionable['ad_group_name'].fillna('').astype(str).str.strip() != '')
        ]
        if actionable.empty:
            return pd.DataFrame()

        actionable['campaign_id'] = actionable['campaign_id'].fillna('').astype(str).str.strip()
        actionable['ad_group_id'] = actionable['ad_group_id'].fillna('').astype(str).str.strip()
        actionable = actionable[
            (actionable['campaign_id'] != '')
            & (actionable['ad_group_id'] != '')
        ]
        if actionable.empty:
            return pd.DataFrame()

        actionable = actionable.drop_duplicates(
            subset=['campaign_name', 'ad_group_name', 'normalized_term'],
            keep='first'
        )

        bulk = pd.DataFrame(index=actionable.index)
        bulk['Product'] = 'Sponsored Brands'
        bulk['Entity'] = 'Negative Keyword'
        bulk['Operation'] = 'Create'
        bulk['Campaign ID'] = actionable['campaign_id']
        bulk['Ad Group ID'] = actionable['ad_group_id']
        bulk['Keyword ID'] = ''
        bulk['Product Targeting ID'] = ''
        bulk['Campaign Name'] = actionable['campaign_name']
        bulk['Ad Group Name'] = actionable['ad_group_name']
        bulk['State'] = 'enabled'
        bulk['Keyword Text'] = actionable['normalized_term']
        bulk['Match Type'] = 'Negative Phrase'
        bulk['Bid'] = ''
        bulk['Budget'] = ''
        bulk['Optimizer Action'] = actionable['search_term_action']
        bulk['ad_type'] = 'SB'
        bulk['source_type'] = 'negative'
        bulk['campaign'] = actionable['campaign_name']
        bulk['ad_group'] = actionable['ad_group_name']
        return bulk.reset_index(drop=True)

    def process(self):
        perf = self._load_search_terms()
        standard_sheet, multi_sheet, wb = self._bulk_sheets()
        perf = self._attach_ids_from_bulk(perf, standard_sheet, multi_sheet)
        existing_state = self._build_existing_state(standard_sheet, multi_sheet)

        bid_updates = []
        budget_updates = []

        perf_rows_loaded = len(perf)
        perf_rows_with_ids = 0 if perf.empty else int(((perf['keyword_id'] != '') | (perf['product_targeting_id'] != '')).sum())
        search_rows_above_floor = 0 if perf.empty else int((perf['clicks'] >= 4).sum())
        search_rows_zero_order_loss = 0 if perf.empty else int((((perf['clicks'] >= 4) | (perf['spend'] >= 12)) & (perf['orders'] == 0)).sum())

        if not perf.empty:
            if standard_sheet is not None and not standard_sheet.empty:
                bid_updates.append(self._build_bid_updates_for_sheet(standard_sheet, perf.copy(), 'Keyword ID'))
                bid_updates.append(self._build_bid_updates_for_sheet(standard_sheet, perf.copy(), 'Product Targeting ID'))
                bid_updates.append(self._build_ad_group_fallback_bid_updates(standard_sheet, perf.copy()))
                budget_updates.append(self._build_campaign_budget_updates(standard_sheet, perf))

            if multi_sheet is not None and not multi_sheet.empty:
                bid_updates.append(self._build_bid_updates_for_sheet(multi_sheet, perf.copy(), 'Keyword ID'))
                bid_updates.append(self._build_bid_updates_for_sheet(multi_sheet, perf.copy(), 'Product Targeting ID'))
                bid_updates.append(self._build_ad_group_fallback_bid_updates(multi_sheet, perf.copy()))
                budget_updates.append(self._build_campaign_budget_updates(multi_sheet, perf))

        search_term_actions = self._build_search_term_actions(perf, existing_state)
        harvest_bulk_updates = self._generate_harvest_bulk_updates(
            search_term_actions,
            standard_sheet=standard_sheet,
            multi_sheet=multi_sheet,
        )
        negative_bulk_updates = self._generate_negative_bulk_updates(
            search_term_actions,
            standard_sheet=standard_sheet,
            multi_sheet=multi_sheet,
        )

        bid_updates_df = safe_concat_frames(bid_updates, ignore_index=True)
        budget_updates_df = safe_concat_frames(budget_updates, ignore_index=True)
        combined = safe_concat_frames(
            [bid_updates_df, harvest_bulk_updates, negative_bulk_updates, budget_updates_df],
            ignore_index=True
        )

        action_log = pd.DataFrame()
        if not search_term_actions.empty:
            keep_cols = ['campaign_name', 'ad_group_name', 'search_term', 'clicks', 'spend', 'sales', 'orders', 'roas', 'search_term_action', 'recommended_bid']
            action_log = search_term_actions[keep_cols].copy()
            action_log['ad_type'] = 'SB'
            action_log['source_type'] = 'search_term'
            action_log['optimizer_action'] = action_log['search_term_action']

        summary = {
            'bid_increases': int((combined.get('optimizer_action', combined.get('Optimizer Action', pd.Series(dtype=str))) == 'INCREASE_BID').sum()) if not combined.empty else 0,
            'bid_decreases': int((combined.get('optimizer_action', combined.get('Optimizer Action', pd.Series(dtype=str))) == 'DECREASE_BID').sum()) if not combined.empty else 0,
            'budget_increases': int((combined.get('optimizer_action', combined.get('Optimizer Action', pd.Series(dtype=str))) == 'INCREASE_BUDGET').sum()) if not combined.empty else 0,
            'budget_decreases': int((combined.get('optimizer_action', combined.get('Optimizer Action', pd.Series(dtype=str))) == 'DECREASE_BUDGET').sum()) if not combined.empty else 0,
            'harvested_keywords': int((search_term_actions.get('search_term_action', pd.Series(dtype=str)) == 'HARVEST_TO_EXACT').sum()) if not search_term_actions.empty else 0,
            'negatives_added': int((search_term_actions.get('search_term_action', pd.Series(dtype=str)) == 'ADD_NEGATIVE_PHRASE').sum()) if not search_term_actions.empty else 0,
        }

        diagnostics = pd.DataFrame([{
            'ad_type': 'SB',
            'perf_rows_loaded': perf_rows_loaded,
            'perf_rows_with_ids': perf_rows_with_ids,
            'search_rows_above_click_floor': search_rows_above_floor,
            'search_rows_zero_order_loss': search_rows_zero_order_loss,
            'harvest_rows_generated': len(harvest_bulk_updates),
            'negative_rows_generated': len(negative_bulk_updates),
            'bid_rows_generated': len(bid_updates_df),
            'budget_rows_generated': len(budget_updates_df),
            'combined_rows_generated': len(combined),
            'used_name_fallback_logic': True,
        }])

        account_summary = {
            'total_spend': round(float(perf['spend'].sum()), 2) if not perf.empty else 0.0,
            'total_sales': round(float(perf['sales'].sum()), 2) if not perf.empty else 0.0,
        }

        return {
            'ad_type': 'SB',
            'combined_bulk_updates': combined,
            'bid_recommendations': bid_updates_df,
            'campaign_budget_actions': budget_updates_df,
            'search_term_actions': action_log,
            'simulation_summary': summary,
            'account_summary': account_summary,
            'campaign_health_dashboard': pd.DataFrame(),
            'smart_warnings': [],
            'optimization_suggestions': [],
            'pre_run_preview': summary,
            'diagnostics': diagnostics,
        }

class SponsoredDisplayOptimizer(_Phase2BaseOptimizer):
    def __init__(self, bulk_file, targeting_file=None, **kwargs):
        super().__init__(bulk_file=bulk_file, enable_search_harvesting=False, enable_negative_keywords=False, **kwargs)
        self.targeting_file = targeting_file

    def _load_targeting(self):
        df = self.load_file(self.targeting_file) if self.targeting_file is not None else None

        if df is None or df.empty:
            wb = self.load_bulk_workbook()
            if 'Sponsored Display Campaigns' in wb:
                df = wb['Sponsored Display Campaigns']
            elif 'RAS Campaigns' in wb:
                df = wb['RAS Campaigns']

        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()

        out['campaign_id'] = self.clean_text(out.get('Campaign ID', pd.Series('', index=out.index)))
        out['ad_group_id'] = self.clean_text(out.get('Ad Group ID', pd.Series('', index=out.index)))
        out['targeting_id'] = self.clean_text(out.get('Targeting ID', out.get('Target ID', pd.Series('', index=out.index))))

        out['campaign_name'] = self.clean_text(
            out.get('Campaign Name', out.get('Campaign Name (Informational only)', pd.Series('', index=out.index)))
        )
        out['ad_group_name'] = self.clean_text(
            out.get('Ad Group Name', out.get('Ad Group Name (Informational only)', pd.Series('', index=out.index)))
        )

        out['target_text'] = self.clean_text(
            out.get('Targeting', out.get('Target', out.get('Resolved Expression', pd.Series('', index=out.index))))
        )

        out['bid'] = self.first_present(out, ['Bid', 'Ad Group Default Bid', 'Default Bid'])
        out['budget'] = self.first_present(out, ['Budget', 'Daily Budget'])
        out['clicks'] = self.first_present(out, ['Clicks'])
        out['spend'] = self.first_present(out, ['Spend', 'Cost'])
        out['sales'] = self.first_present(out, ['Sales', 'Attributed Sales 14 Day', 'Attributed Sales'])
        out['orders'] = self.first_present(out, ['Orders', 'Purchases', 'Attributed Conversions 14 Day'])
        out['roas'] = self.first_present(out, ['ROAS', 'Return on Ad Spend'])

        if 'ROAS' not in out.columns and 'Return on Ad Spend' not in out.columns:
            out['roas'] = np.where(out['spend'] > 0, out['sales'] / out['spend'], 0)

        out['tactic'] = self.clean_text(out.get('Tactic', out.get('Targeting Type', pd.Series('', index=out.index))))

        out['campaign_name_key'] = self.normalize_join_text(out['campaign_name'])
        out['ad_group_name_key'] = self.normalize_join_text(out['ad_group_name'])
        out['target_text_key'] = self.normalize_join_text(out['target_text'])

        return out

    def process(self):
        perf = self._load_targeting()

        if perf.empty:
            return {
                'ad_type': 'SD',
                'combined_bulk_updates': pd.DataFrame(),
                'bid_recommendations': pd.DataFrame(),
                'campaign_budget_actions': pd.DataFrame(),
                'search_term_actions': pd.DataFrame(),
                'simulation_summary': {'bid_increases':0,'bid_decreases':0,'budget_increases':0,'budget_decreases':0,'harvested_keywords':0,'negatives_added':0},
                'account_summary': {'total_spend':0.0,'total_sales':0.0},
                'campaign_health_dashboard': pd.DataFrame(),
                'smart_warnings': [],
                'optimization_suggestions': [],
                'pre_run_preview': {},
                'diagnostics': pd.DataFrame([{
                    'ad_type': 'SD',
                    'perf_rows_loaded': 0,
                    'perf_rows_with_targeting_ids': 0,
                    'rows_above_click_floor': 0,
                    'rows_zero_order_loss': 0,
                    'bid_rows_generated': 0,
                    'budget_rows_generated': 0,
                    'combined_rows_generated': 0,
                    'used_campaign_or_adgroup_fallback_logic': True,
                }]),
            }

        wb = self.load_bulk_workbook()
        sd_name = find_matching_sheet_name(
            wb.keys(),
            ['Sponsored Display Campaigns', 'Sponsored Display', 'SD Campaigns', 'SD']
        )
        ras_name = find_matching_sheet_name(
            wb.keys(),
            ['RAS Campaigns', 'RAS']
        )
        sd_sheet = wb.get(sd_name) if sd_name else None
        ras_sheet = wb.get(ras_name) if ras_name else None

        bid_updates = []
        budget_updates = []

        perf_rows_with_targeting_ids = int((perf['targeting_id'] != '').sum())
        rows_above_click_floor = int((perf['clicks'] >= 4).sum())
        rows_zero_order_loss = int((((perf['clicks'] >= 4) | (perf['spend'] >= 12)) & (perf['orders'] == 0)).sum())

        def sd_bid_action(row):
            clicks = float(row['clicks'] or 0)
            orders = float(row['orders'] or 0)
            roas = float(row['roas'] or 0)
            spend = float(row['spend'] or 0)

            if (clicks >= 4 or spend >= 12) and orders == 0:
                return 'DECREASE_BID'
            if clicks >= 4 and roas > 0 and roas < self.min_roas:
                return 'DECREASE_BID'
            if orders >= self.min_orders_for_scaling and roas >= self.min_roas * 1.10:
                return 'INCREASE_BID'
            return None

        def sd_budget_action(row):
            orders = float(row['orders'] or 0)
            roas = float(row['roas'] or 0)
            spend = float(row['spend'] or 0)

            if spend >= 15 and orders == 0:
                return 'DECREASE_BUDGET'
            if roas > 0 and roas < self.min_roas:
                return 'DECREASE_BUDGET'
            if orders >= self.min_orders_for_scaling and roas >= self.min_roas * 1.10:
                return 'INCREASE_BUDGET'
            return None

        for sheet, id_col, budget_col in [
            (sd_sheet, 'Targeting ID', 'Budget'),
            (ras_sheet, 'Target ID', 'Budget'),
        ]:
            if sheet is None or sheet.empty:
                continue

            work = sheet.copy()
            work['_campaign_key'] = self.clean_text(work.get('Campaign ID', pd.Series('', index=work.index)))
            work['_ad_group_key'] = self.clean_text(work.get('Ad Group ID', pd.Series('', index=work.index)))
            work['_target_key'] = self.clean_text(work.get(id_col, pd.Series('', index=work.index)))
            work['_campaign_name_key'] = self.normalize_join_text(
                work.get('Campaign Name', work.get('Campaign Name (Informational only)', pd.Series('', index=work.index)))
            )
            work['_ad_group_name_key'] = self.normalize_join_text(
                work.get('Ad Group Name', work.get('Ad Group Name (Informational only)', pd.Series('', index=work.index)))
            )

            target_text_col = None
            for c in ['Targeting', 'Target', 'Resolved Expression', 'Expression']:
                if c in work.columns:
                    target_text_col = c
                    break

            if target_text_col is not None:
                work['_target_text_key'] = self.normalize_join_text(work[target_text_col])
            else:
                work['_target_text_key'] = ''

            if 'Entity' in work.columns:
                target_rows = work[work['Entity'].astype(str).str.contains('Target|Audience|Product', case=False, na=False)].copy()
            else:
                target_rows = work.copy()

            if not target_rows.empty and 'Bid' in target_rows.columns:
                perf_by_id = perf[perf['targeting_id'] != ''].groupby('targeting_id', as_index=False).agg(
                    clicks=('clicks','sum'),
                    spend=('spend','sum'),
                    sales=('sales','sum'),
                    orders=('orders','sum'),
                    roas=('roas','mean'),
                )
                perf_by_id['_target_key'] = self.clean_text(perf_by_id['targeting_id'])

                joined = target_rows.merge(perf_by_id, on='_target_key', how='left')

                perf_fallback = perf.groupby(
                    ['campaign_name_key', 'ad_group_name_key', 'target_text_key'],
                    as_index=False
                ).agg(
                    fb_clicks=('clicks','sum'),
                    fb_spend=('spend','sum'),
                    fb_sales=('sales','sum'),
                    fb_orders=('orders','sum'),
                    fb_roas=('roas','mean'),
                )

                joined = joined.merge(
                    perf_fallback,
                    left_on=['_campaign_name_key', '_ad_group_name_key', '_target_text_key'],
                    right_on=['campaign_name_key', 'ad_group_name_key', 'target_text_key'],
                    how='left'
                )

                perf_fallback_ad_group = perf.groupby(
                    ['campaign_name_key', 'ad_group_name_key'],
                    as_index=False
                ).agg(
                    ag_clicks=('clicks','sum'),
                    ag_spend=('spend','sum'),
                    ag_sales=('sales','sum'),
                    ag_orders=('orders','sum'),
                    ag_roas=('roas','mean'),
                )

                joined = joined.merge(
                    perf_fallback_ad_group,
                    left_on=['_campaign_name_key', '_ad_group_name_key'],
                    right_on=['campaign_name_key', 'ad_group_name_key'],
                    how='left'
                )

                joined['clicks'] = joined['clicks'].fillna(joined['fb_clicks']).fillna(joined['ag_clicks']).fillna(0)
                joined['spend'] = joined['spend'].fillna(joined['fb_spend']).fillna(joined['ag_spend']).fillna(0)
                joined['sales'] = joined['sales'].fillna(joined['fb_sales']).fillna(joined['ag_sales']).fillna(0)
                joined['orders'] = joined['orders'].fillna(joined['fb_orders']).fillna(joined['ag_orders']).fillna(0)
                joined['roas'] = joined['roas'].fillna(joined['fb_roas']).fillna(joined['ag_roas']).fillna(0)

                joined['current_bid'] = self.safe_numeric(joined['Bid'])
                joined['optimizer_action'] = joined.apply(
                    lambda r: sd_bid_action(r) if self.enable_bid_updates else None,
                    axis=1,
                )
                joined = joined[joined['optimizer_action'].notna()].copy()

                if not joined.empty:
                    joined['new_bid'] = joined.apply(lambda r: self._adjust_bid(r['current_bid'], r['optimizer_action']), axis=1)
                    joined = joined[joined['new_bid'] != joined['current_bid']].copy()

                    if not joined.empty:
                        joined['Bid'] = joined['new_bid']
                        joined['Operation'] = 'update'
                        joined['Product'] = 'Sponsored Display'
                        joined['ad_type'] = 'SD'
                        joined['source_type'] = 'bid'
                        joined['campaign'] = joined.get('Campaign Name', joined.get('campaign_name', ''))
                        joined['ad_group'] = joined.get('Ad Group Name', joined.get('ad_group_name', ''))
                        bid_updates.append(joined)

            if budget_col in work.columns and 'Campaign ID' in work.columns:
                campaigns = work.copy()
                if 'Entity' in campaigns.columns:
                    campaigns = campaigns[campaigns['Entity'].astype(str).str.contains('Campaign', case=False, na=False)].copy()

                if not campaigns.empty:
                    campaigns['_campaign_key'] = self.clean_text(campaigns['Campaign ID'])
                    campaigns['campaign_name_key'] = self.normalize_join_text(
                        campaigns.get('Campaign Name', campaigns.get('Campaign Name (Informational only)', pd.Series('', index=campaigns.index)))
                    )

                    if (perf['campaign_id'] != '').any():
                        perf_campaigns = perf.groupby('campaign_id', as_index=False).agg(
                            clicks=('clicks','sum'),
                            spend=('spend','sum'),
                            sales=('sales','sum'),
                            orders=('orders','sum'),
                            roas=('roas','mean'),
                        )
                        perf_campaigns['_campaign_key'] = self.clean_text(perf_campaigns['campaign_id'])
                        joined = campaigns.merge(perf_campaigns, on='_campaign_key', how='left')
                    else:
                        perf_campaigns = perf.groupby('campaign_name_key', as_index=False).agg(
                            clicks=('clicks','sum'),
                            spend=('spend','sum'),
                            sales=('sales','sum'),
                            orders=('orders','sum'),
                            roas=('roas','mean'),
                        )

                        joined = campaigns.merge(
                            perf_campaigns,
                            on='campaign_name_key',
                            how='left'
                        )

                    joined['clicks'] = joined['clicks'].fillna(0)
                    joined['spend'] = joined['spend'].fillna(0)
                    joined['orders'] = joined['orders'].fillna(0)
                    joined['roas'] = joined['roas'].fillna(0)
                    joined['current_budget'] = self.safe_numeric(joined[budget_col])

                    joined['optimizer_action'] = joined.apply(
                        lambda r: sd_budget_action(r) if self.enable_budget_updates else None,
                        axis=1,
                    )
                    joined = joined[joined['optimizer_action'].notna()].copy()

                    if not joined.empty:
                        joined['new_budget'] = joined.apply(lambda r: self._adjust_budget(r['current_budget'], r['optimizer_action']), axis=1)
                        joined = joined[joined['new_budget'] != joined['current_budget']].copy()

                        if not joined.empty:
                            joined[budget_col] = joined['new_budget']
                            joined['Operation'] = 'update'
                            joined['Product'] = 'Sponsored Display'
                            joined['ad_type'] = 'SD'
                            joined['source_type'] = 'budget'
                            joined['campaign'] = joined.get('Campaign Name', joined.get('campaign_name', ''))
                            joined['ad_group'] = ''
                            budget_updates.append(joined)

        bid_updates_df = safe_concat_frames(bid_updates, ignore_index=True)
        budget_updates_df = safe_concat_frames(budget_updates, ignore_index=True)
        combined = safe_concat_frames([bid_updates_df, budget_updates_df], ignore_index=True)

        summary = {
            'bid_increases': int((combined.get('optimizer_action', pd.Series(dtype=str)) == 'INCREASE_BID').sum()) if not combined.empty else 0,
            'bid_decreases': int((combined.get('optimizer_action', pd.Series(dtype=str)) == 'DECREASE_BID').sum()) if not combined.empty else 0,
            'budget_increases': int((combined.get('optimizer_action', pd.Series(dtype=str)) == 'INCREASE_BUDGET').sum()) if not combined.empty else 0,
            'budget_decreases': int((combined.get('optimizer_action', pd.Series(dtype=str)) == 'DECREASE_BUDGET').sum()) if not combined.empty else 0,
            'harvested_keywords': 0,
            'negatives_added': 0,
        }

        diagnostics = pd.DataFrame([{
            'ad_type': 'SD',
            'perf_rows_loaded': len(perf),
            'perf_rows_with_targeting_ids': perf_rows_with_targeting_ids,
            'rows_above_click_floor': rows_above_click_floor,
            'rows_zero_order_loss': rows_zero_order_loss,
            'bid_rows_generated': len(bid_updates_df),
            'budget_rows_generated': len(budget_updates_df),
            'combined_rows_generated': len(combined),
            'used_campaign_or_adgroup_fallback_logic': True,
        }])

        account_summary = {
            'total_spend': round(float(perf['spend'].sum()), 2),
            'total_sales': round(float(perf['sales'].sum()), 2),
        }

        return {
            'ad_type': 'SD',
            'combined_bulk_updates': combined,
            'bid_recommendations': bid_updates_df,
            'campaign_budget_actions': budget_updates_df,
            'search_term_actions': pd.DataFrame(),
            'simulation_summary': summary,
            'account_summary': account_summary,
            'campaign_health_dashboard': pd.DataFrame(),
            'smart_warnings': [],
            'optimization_suggestions': [],
            'pre_run_preview': summary,
            'diagnostics': diagnostics,
        }

class Phase2AdsOrchestrator:
    def __init__(
        self,
        bulk_file,
        business_report_file=None,
        sqp_report_file=None,
        sp_search_term_file=None,
        sp_targeting_file=None,
        sp_impression_share_file=None,
        sb_search_term_file=None,
        sb_impression_share_file=None,
        sd_targeting_file=None,
        min_roas=3.0,
        min_clicks=8,
        zero_order_click_threshold=12,
        zero_order_action='Both',
        strategy_mode='Balanced',
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
        tacos_constrained=False,
    ):
        self.kwargs = locals().copy()
        self.kwargs.pop('self')

        self.global_spend_summary = self._compute_global_spend_summary()
        self.global_total_ad_spend = float(self.global_spend_summary.get('total_spend', 0) or 0)
        self.global_business_total_sales = self._compute_business_total_sales()

        self.global_tacos_pct = None
        self.global_tacos_constrained = False

        if (
            self.kwargs.get('enable_tacos_control')
            and self.global_business_total_sales is not None
            and self.global_business_total_sales > 0
        ):
            self.global_tacos_pct = round((self.global_total_ad_spend / self.global_business_total_sales) * 100, 2)
            self.global_tacos_constrained = self.global_tacos_pct > float(self.kwargs.get('max_tacos_target', 15.0))

    def _clone_file_obj(self, file_obj):
        if file_obj is None:
            return None
        if isinstance(file_obj, (str, bytes, os.PathLike)):
            return file_obj
        if isinstance(file_obj, io.BytesIO):
            return io.BytesIO(file_obj.getvalue())
        if hasattr(file_obj, 'getvalue'):
            return io.BytesIO(file_obj.getvalue())
        if hasattr(file_obj, 'read'):
            try:
                pos = file_obj.tell()
            except Exception:
                pos = None
            try:
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                data = file_obj.read()
            finally:
                try:
                    if pos is not None and hasattr(file_obj, 'seek'):
                        file_obj.seek(pos)
                except Exception:
                    pass
            return io.BytesIO(data)
        raise ValueError("Unsupported file object type.")

    def _compute_global_spend_summary(self):
        validator = Phase2UploadValidator(
            bulk_file=self.kwargs['bulk_file'],
            business_report_file=self.kwargs['business_report_file'],
            sqp_report_file=self.kwargs['sqp_report_file'],
            sp_search_term_file=self.kwargs['sp_search_term_file'],
            sp_targeting_file=self.kwargs['sp_targeting_file'],
            sp_impression_share_file=self.kwargs['sp_impression_share_file'],
            sb_search_term_file=self.kwargs['sb_search_term_file'],
            sb_impression_share_file=self.kwargs['sb_impression_share_file'],
            sd_targeting_file=self.kwargs['sd_targeting_file'],
        )
        analysis = validator.analyze()
        return analysis.get('spend_summary', {}) if isinstance(analysis, dict) else {}

    def _compute_business_total_sales(self):
        business_file = self.kwargs.get('business_report_file')
        if business_file is None:
            return None
    
        original_name = getattr(business_file, 'name', '') if not isinstance(business_file, (str, os.PathLike)) else str(business_file)
        ext = os.path.splitext(original_name)[1].lower()
    
        cloned = self._clone_file_obj(business_file)
    
        if ext == '.csv':
            df = pd.read_csv(cloned)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(cloned, engine='openpyxl')
        else:
            # fallback: try csv first, then excel
            try:
                cloned.seek(0)
                df = pd.read_csv(cloned)
            except Exception:
                cloned.seek(0)
                df = pd.read_excel(cloned, engine='openpyxl')
    
        df.columns = [str(c).strip() for c in df.columns]
    
        def _clean_money(series):
            return pd.to_numeric(
                series.astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("(", "-", regex=False)
                .str.replace(")", "", regex=False)
                .str.strip(),
                errors="coerce",
            ).fillna(0)
    
        exact_ordered_cols = [
            c for c in df.columns
            if str(c).strip().lower() == "ordered product sales"
        ]
        if exact_ordered_cols:
            return round(sum(float(_clean_money(df[col]).sum()) for col in exact_ordered_cols), 2)
    
        ordered_component_cols = [
            c for c in df.columns
            if str(c).strip().lower() in {
                "ordered product sales - b2c",
                "ordered product sales - b2b",
            }
        ]
        if ordered_component_cols:
            return round(sum(float(_clean_money(df[col]).sum()) for col in ordered_component_cols), 2)
    
        fallback_cols = [
            c for c in df.columns
            if str(c).strip().lower() in {"total sales", "sales"}
        ]
        if fallback_cols:
            return round(float(_clean_money(df[fallback_cols[0]]).sum()), 2)
    
        return None

    def _sp_engine(self):
        return AdsOptimizerEngine(
            bulk_file=self.kwargs['bulk_file'],
            search_term_file=self.kwargs['sp_search_term_file'],
            targeting_file=self.kwargs['sp_targeting_file'],
            impression_share_file=self.kwargs['sp_impression_share_file'],
            business_report_file=self.kwargs['business_report_file'],
            sqp_report_file=self.kwargs['sqp_report_file'],
            min_roas=self.kwargs['min_roas'],
            min_clicks=self.kwargs['min_clicks'],
            zero_order_click_threshold=self.kwargs['zero_order_click_threshold'],
            zero_order_action=self.kwargs['zero_order_action'],
            strategy_mode=self.kwargs['strategy_mode'],
            enable_bid_updates=self.kwargs['enable_bid_updates'],
            enable_search_harvesting=self.kwargs['enable_search_harvesting'],
            enable_negative_keywords=self.kwargs['enable_negative_keywords'],
            enable_budget_updates=self.kwargs['enable_budget_updates'],
            enable_tacos_control=self.kwargs['enable_tacos_control'],
            max_tacos_target=self.kwargs['max_tacos_target'],
            enable_monthly_budget_control=self.kwargs['enable_monthly_budget_control'],
            monthly_account_budget=self.kwargs['monthly_account_budget'],
            month_to_date_spend=self.kwargs['month_to_date_spend'],
            pacing_buffer_pct=self.kwargs['pacing_buffer_pct'],
            max_bid_cap=self.kwargs['max_bid_cap'],
            max_budget_cap=self.kwargs['max_budget_cap'],
            external_total_ad_spend=self.global_total_ad_spend,
            tacos_constrained_override=self.global_tacos_constrained,
        )

    def _sb_engine(self):
        return SponsoredBrandsOptimizer(
            bulk_file=self.kwargs['bulk_file'],
            search_term_file=self.kwargs['sb_search_term_file'],
            impression_share_file=self.kwargs['sb_impression_share_file'],
            min_roas=self.kwargs['min_roas'],
            min_clicks=self.kwargs['min_clicks'],
            zero_order_click_threshold=self.kwargs['zero_order_click_threshold'],
            zero_order_action=self.kwargs['zero_order_action'],
            strategy_mode=self.kwargs['strategy_mode'],
            enable_bid_updates=self.kwargs['enable_bid_updates'],
            enable_search_harvesting=self.kwargs['enable_search_harvesting'],
            enable_negative_keywords=self.kwargs['enable_negative_keywords'],
            enable_budget_updates=self.kwargs['enable_budget_updates'],
            max_bid_cap=self.kwargs['max_bid_cap'],
            max_budget_cap=self.kwargs['max_budget_cap'],
            tacos_constrained=self.global_tacos_constrained,
        )

    def _sd_engine(self):
        return SponsoredDisplayOptimizer(
            bulk_file=self.kwargs['bulk_file'],
            targeting_file=self.kwargs['sd_targeting_file'],
            min_roas=self.kwargs['min_roas'],
            min_clicks=self.kwargs['min_clicks'],
            zero_order_click_threshold=self.kwargs['zero_order_click_threshold'],
            zero_order_action=self.kwargs['zero_order_action'],
            strategy_mode=self.kwargs['strategy_mode'],
            enable_bid_updates=self.kwargs['enable_bid_updates'],
            enable_budget_updates=self.kwargs['enable_budget_updates'],
            max_bid_cap=self.kwargs['max_bid_cap'],
            max_budget_cap=self.kwargs['max_budget_cap'],
            tacos_constrained=self.global_tacos_constrained,
        )

    def analyze(self):
        return self.process()

    def process(self):
        validator = Phase2UploadValidator(
            bulk_file=self.kwargs['bulk_file'],
            business_report_file=self.kwargs['business_report_file'],
            sqp_report_file=self.kwargs['sqp_report_file'],
            sp_search_term_file=self.kwargs['sp_search_term_file'],
            sp_targeting_file=self.kwargs['sp_targeting_file'],
            sp_impression_share_file=self.kwargs['sp_impression_share_file'],
            sb_search_term_file=self.kwargs['sb_search_term_file'],
            sb_impression_share_file=self.kwargs['sb_impression_share_file'],
            sd_targeting_file=self.kwargs['sd_targeting_file'],
        )
        validation = validator.analyze()
        runnable = validation.get('runnable_types', [])
        per_type = {}
        execution_summary_rows = []
        all_bulk_updates = []
        all_bid_updates = []
        all_search_actions = []
        all_budget_actions = []
        all_warnings = []
        all_suggestions = []
        optimizer_diagnostics_frames = []
        sqp_opportunities = pd.DataFrame()
        sqp_summary = {}
        sp_account_health = {}
        sp_campaign_dashboard = pd.DataFrame()
        run_history = pd.DataFrame()
        total_preview = {'bid_increases':0,'bid_decreases':0,'budget_increases':0,'budget_decreases':0,'harvested_keywords':0,'negatives_added':0}
        total_summary = {'bid_increases':0,'bid_decreases':0,'budget_increases':0,'budget_decreases':0,'harvested_keywords':0,'negatives_added':0}
        combined_account_summary = {'total_spend':0.0,'total_sales':0.0,'campaigns_under_target':0,'campaigns_scalable':0,'campaigns_waste_alert':0}

        if 'SP' in runnable:
            sp_out = self._sp_engine().process()
            per_type['SP'] = sp_out
            if isinstance(sp_out.get('diagnostics'), pd.DataFrame) and not sp_out.get('diagnostics').empty:
                optimizer_diagnostics_frames.append(sp_out.get('diagnostics'))
            execution_summary_rows.append({'ad_type':'SP','optimized':True,'status':'Optimized','actions':len(sp_out.get('combined_bulk_updates', pd.DataFrame())), 'spend':sp_out.get('account_summary',{}).get('total_spend',0.0)})
            for key, target in [('combined_bulk_updates', all_bulk_updates), ('bid_recommendations', all_bid_updates), ('search_term_actions', all_search_actions), ('campaign_budget_actions', all_budget_actions)]:
                df = sp_out.get(key)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if 'ad_type' not in df.columns:
                        df = df.copy(); df['ad_type'] = 'SP'
                    target.append(df)
            all_warnings.extend(sp_out.get('smart_warnings', []))
            all_suggestions.extend(sp_out.get('optimization_suggestions', []))
            sqp_opportunities = sp_out.get('sqp_opportunities', pd.DataFrame())
            sqp_summary = sp_out.get('sqp_summary', {})
            sp_account_health = sp_out.get('account_health', {})
            sp_campaign_dashboard = sp_out.get('campaign_health_dashboard', pd.DataFrame())
            run_history = sp_out.get('run_history', pd.DataFrame())
            for k in total_preview:
                total_preview[k] += int(sp_out.get('pre_run_preview', {}).get(k, 0) or 0)
                total_summary[k] += int(sp_out.get('simulation_summary', {}).get(k, 0) or 0)
            for k,v in sp_out.get('account_summary', {}).items():
                if isinstance(v, (int,float,np.integer,np.floating)):
                    combined_account_summary[k] = combined_account_summary.get(k,0)+v
        else:
            execution_summary_rows.append({'ad_type':'SP','optimized':False,'status':'Skipped - missing required uploads','actions':0,'spend':validation.get('spend_summary',{}).get('sp_spend',0.0)})

        if 'SB' in runnable:
            sb_out = self._sb_engine().process()
            per_type['SB'] = sb_out
            if isinstance(sb_out.get('diagnostics'), pd.DataFrame) and not sb_out.get('diagnostics').empty:
                optimizer_diagnostics_frames.append(sb_out.get('diagnostics'))
            execution_summary_rows.append({'ad_type':'SB','optimized':True,'status':'Optimized','actions':len(sb_out.get('combined_bulk_updates', pd.DataFrame())), 'spend':sb_out.get('account_summary',{}).get('total_spend',0.0)})
            for key, target in [('combined_bulk_updates', all_bulk_updates), ('bid_recommendations', all_bid_updates), ('search_term_actions', all_search_actions), ('campaign_budget_actions', all_budget_actions)]:
                df = sb_out.get(key)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if 'ad_type' not in df.columns:
                        df = df.copy(); df['ad_type'] = 'SB'
                    target.append(df)
            for k in total_preview:
                total_preview[k] += int(sb_out.get('pre_run_preview', {}).get(k, 0) or 0)
                total_summary[k] += int(sb_out.get('simulation_summary', {}).get(k, 0) or 0)
            for k,v in sb_out.get('account_summary', {}).items():
                if isinstance(v, (int,float,np.integer,np.floating)):
                    combined_account_summary[k] = combined_account_summary.get(k,0)+v
            all_suggestions.append('Sponsored Brands optimizer used bid, budget, harvest, and negative logic from search term performance.')
            if sb_out.get('diagnostics') is not None and not safe_concat_frames([sb_out.get('diagnostics')]).empty:
                sb_diag = sb_out.get('diagnostics').iloc[0]
                if int(sb_diag.get('combined_rows_generated', 0) or 0) == 0 and int(sb_diag.get('search_rows_zero_order_loss', 0) or 0) > 0:
                    all_warnings.append('Sponsored Brands had inefficient search-term rows but no final actions; review column mapping or campaign structure if this persists.')
        else:
            execution_summary_rows.append({'ad_type':'SB','optimized':False,'status':'Skipped - missing required uploads','actions':0,'spend':validation.get('spend_summary',{}).get('sb_spend',0.0)})

        if 'SD' in runnable:
            sd_out = self._sd_engine().process()
            per_type['SD'] = sd_out
            if isinstance(sd_out.get('diagnostics'), pd.DataFrame) and not sd_out.get('diagnostics').empty:
                optimizer_diagnostics_frames.append(sd_out.get('diagnostics'))
            execution_summary_rows.append({'ad_type':'SD','optimized':True,'status':'Optimized','actions':len(sd_out.get('combined_bulk_updates', pd.DataFrame())), 'spend':sd_out.get('account_summary',{}).get('total_spend',0.0)})
            for key, target in [('combined_bulk_updates', all_bulk_updates), ('bid_recommendations', all_bid_updates), ('search_term_actions', all_search_actions), ('campaign_budget_actions', all_budget_actions)]:
                df = sd_out.get(key)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if 'ad_type' not in df.columns:
                        df = df.copy(); df['ad_type'] = 'SD'
                    target.append(df)
            for k in total_preview:
                total_preview[k] += int(sd_out.get('pre_run_preview', {}).get(k, 0) or 0)
                total_summary[k] += int(sd_out.get('simulation_summary', {}).get(k, 0) or 0)
            for k,v in sd_out.get('account_summary', {}).items():
                if isinstance(v, (int,float,np.integer,np.floating)):
                    combined_account_summary[k] = combined_account_summary.get(k,0)+v
            all_suggestions.append('Sponsored Display optimizer used target-level and campaign budget logic.')
        else:
            execution_summary_rows.append({'ad_type':'SD','optimized':False,'status':'Skipped - missing required uploads','actions':0,'spend':validation.get('spend_summary',{}).get('sd_spend',0.0)})

        execution_summary = pd.DataFrame(execution_summary_rows)
        optimizer_diagnostics = safe_concat_frames(optimizer_diagnostics_frames, ignore_index=True)
        combined_bulk_updates = apply_cross_type_bulk_safeguards(safe_concat_frames(all_bulk_updates, ignore_index=True))
        bid_recommendations = safe_concat_frames(all_bid_updates, ignore_index=True)
        search_term_actions = safe_concat_frames(all_search_actions, ignore_index=True)
        campaign_budget_actions = safe_concat_frames(all_budget_actions, ignore_index=True)

        exported_frames = {
            'execution_summary': execution_summary,
            'optimizer_diagnostics': optimizer_diagnostics,
            'combined_bulk_updates': combined_bulk_updates,
            'bid_recommendations': bid_recommendations,
            'search_term_actions': search_term_actions,
            'campaign_budget_actions': campaign_budget_actions,
        }
        for _name, _df in exported_frames.items():
            if isinstance(_df, pd.DataFrame) and not _df.empty:
                exported_frames[_name] = _dedupe_and_strip_columns(_df)

        execution_summary = exported_frames['execution_summary']
        optimizer_diagnostics = exported_frames['optimizer_diagnostics']
        combined_bulk_updates = exported_frames['combined_bulk_updates']
        bid_recommendations = ensure_trend_columns(ensure_score_column(exported_frames['bid_recommendations']))
        search_term_actions = ensure_trend_columns(ensure_score_column(exported_frames['search_term_actions']))
        campaign_budget_actions = ensure_trend_columns(ensure_score_column(exported_frames['campaign_budget_actions']))
        spend_summary = validation.get('spend_summary', {})
        combined_account_summary['sp_total_spend'] = spend_summary.get('sp_spend', 0.0)
        combined_account_summary['sb_total_spend'] = spend_summary.get('sb_spend', 0.0)
        combined_account_summary['sd_total_spend'] = spend_summary.get('sd_spend', 0.0)
        combined_account_summary['total_spend'] = spend_summary.get('total_spend', combined_account_summary.get('total_spend', 0.0))
        combined_account_summary['sp_total_sales'] = spend_summary.get('sp_sales', 0.0)
        combined_account_summary['sb_total_sales'] = spend_summary.get('sb_sales', 0.0)
        combined_account_summary['sd_total_sales'] = spend_summary.get('sd_sales', 0.0)
        combined_account_summary['total_sales'] = spend_summary.get('total_sales', combined_account_summary.get('total_sales', 0.0))
        if not runnable:
            all_warnings.append('No ad types were ready to optimize. Use the readiness table to fill the missing uploads.')
        return {
            'combined_bulk_updates': combined_bulk_updates,
            'bid_recommendations': bid_recommendations,
            'search_term_actions': search_term_actions,
            'campaign_budget_actions': campaign_budget_actions,
            'account_health': sp_account_health,
            'simulation_summary': total_summary,
            'campaign_health_dashboard': sp_campaign_dashboard,
            'smart_warnings': all_warnings,
            'optimization_suggestions': all_suggestions,
            'pre_run_preview': total_preview,
            'account_summary': combined_account_summary,
            'run_history': run_history,
            'sqp_opportunities': sqp_opportunities,
            'sqp_summary': sqp_summary,
            'execution_summary': execution_summary,
            'validation': validation,
            'per_type_outputs': per_type,
            'runnable_types': runnable,
            'optimizer_diagnostics': optimizer_diagnostics,
        }
