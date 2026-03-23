import io
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def clone_file_obj(file_obj):
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


def get_file_extension(file_obj, fallback: Optional[str] = None):
    if file_obj is None:
        return None

    if isinstance(file_obj, (str, os.PathLike)):
        return os.path.splitext(str(file_obj))[1].lower()

    name = getattr(file_obj, "name", None)
    if isinstance(name, str) and name.strip():
        return os.path.splitext(name)[1].lower()

    return fallback


def load_file(file_obj, expected_ext: Optional[str] = None):
    if file_obj is None:
        return None

    ext = get_file_extension(file_obj, fallback=expected_ext)
    cloned = clone_file_obj(file_obj)

    if ext == ".csv":
        return pd.read_csv(cloned)

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(cloned, engine="openpyxl")

    raise ValueError(f"Unsupported file type: {ext}")


def load_excel_sheet(file_obj, sheet_name: str, dtype=None):
    cloned = clone_file_obj(file_obj)
    return pd.read_excel(
        cloned,
        sheet_name=sheet_name,
        engine="openpyxl",
        dtype=dtype,
    )


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def clean_text(series):
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned.replace("nan", "")
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    return cleaned


def clean_percent_series(series):
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def clean_currency_series(series):
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def get_first_existing_column(df: pd.DataFrame, candidates: Iterable[str], label: str):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Column for {label} not found. Tried {list(candidates)}. Available: {list(df.columns)}"
    )


def get_optional_column(df: pd.DataFrame, candidates: Iterable[str]):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def combine_preferred_columns(
    df: pd.DataFrame,
    primary_candidates: Iterable[str],
    fallback_candidates: Iterable[str],
    label: str,
):
    primary_col = get_optional_column(df, primary_candidates)
    fallback_col = get_optional_column(df, fallback_candidates)

    if primary_col is None and fallback_col is None:
        raise KeyError(
            f"Could not find a valid column for {label}. "
            f"Tried primary={list(primary_candidates)}, fallback={list(fallback_candidates)}. "
            f"Available: {list(df.columns)}"
        )

    if primary_col is not None:
        primary_series = clean_text(df[primary_col]).replace("", np.nan)
    else:
        primary_series = pd.Series([np.nan] * len(df), index=df.index)

    if fallback_col is not None:
        fallback_series = clean_text(df[fallback_col]).replace("", np.nan)
    else:
        fallback_series = pd.Series([np.nan] * len(df), index=df.index)

    return primary_series.combine_first(fallback_series).fillna("")


def blank_series(df: pd.DataFrame):
    return pd.Series([""] * len(df), index=df.index)


def percent_to_ratio(value: float) -> float:
    try:
        return float(value) / 100.0
    except Exception:
        return 0.0


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.where(denominator != 0, numerator / denominator, 0.0)


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "clicks" not in out.columns:
        out["clicks"] = 0.0
    if "impressions" not in out.columns:
        out["impressions"] = 0.0
    if "spend" not in out.columns:
        out["spend"] = 0.0
    if "sales" not in out.columns:
        out["sales"] = 0.0
    if "orders" not in out.columns:
        out["orders"] = 0.0

    out["ctr"] = safe_divide(out["clicks"], out["impressions"])
    out["cpc"] = safe_divide(out["spend"], out["clicks"])
    out["cvr"] = safe_divide(out["orders"], out["clicks"])
    out["roas"] = safe_divide(out["sales"], out["spend"])
    out["acos"] = safe_divide(out["spend"], out["sales"])

    return out


def normalize_key(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def ensure_columns(df: pd.DataFrame, columns: list[str], default="") -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = default
    return out


def coerce_string_id_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    return out


def to_excel_bytes_multi(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = str(sheet_name)[:31] if str(sheet_name).strip() else "Sheet1"
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_excel(
                writer,
                index=False,
                sheet_name=safe_name,
            )
    return output.getvalue()