# tft_utils.py

import pandas as pd
import numpy as np

def _to_naive_utc(ts):
    return pd.to_datetime(ts, utc=True).dt.tz_convert(None)

def make_series(df: pd.DataFrame, ts_col: str, value_col: str, series_name: str):
    """
    Convert a (ts, value) dataframe to standard columns:
      ts_utc (naive), value (float), series (str)
    """
    out = df[[ts_col, value_col]].copy()
    out = out.rename(columns={ts_col: "ts_utc", value_col: "value"})
    out["ts_utc"] = _to_naive_utc(out["ts_utc"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["ts_utc", "value"])
    out["series"] = series_name
    return out[["ts_utc", "value", "series"]]

def melt_wide_to_long(df_wide: pd.DataFrame):
    """
    For a 'wide' appliance table: first column is timestamp, remaining are appliances.
    Returns long df: ts_utc, appliance, usage_kwh
    """
    if df_wide.shape[1] < 2:
        return pd.DataFrame(columns=["ts_utc", "appliance", "usage_kwh"])
    ts_col = df_wide.columns[0]
    value_cols = [c for c in df_wide.columns if c != ts_col]
    long_df = df_wide.melt(id_vars=[ts_col], value_vars=value_cols,
                           var_name="appliance", value_name="usage_kwh")
    long_df = long_df.rename(columns={ts_col: "ts_utc"})
    long_df["ts_utc"] = _to_naive_utc(long_df["ts_utc"])
    long_df["usage_kwh"] = pd.to_numeric(long_df["usage_kwh"], errors="coerce")
    return long_df.dropna(subset=["ts_utc", "usage_kwh"])

def prepare_time_series_df(df: pd.DataFrame, target_col: str = "value", group_col: str = "series"):
    """
    Prepare dataframe for PyTorch Forecasting TimeSeriesDataSet.

    Expects columns: ts_utc (naive), <target_col>, <group_col>
    Adds per-group time_idx + calendar covariates.
    """
    x = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(x["ts_utc"]):
        x["ts_utc"] = _to_naive_utc(x["ts_utc"])
    x = x.sort_values([group_col, "ts_utc"]).dropna(subset=["ts_utc", target_col]).reset_index(drop=True)

    # time index per group (must increase by 1 within each group)
    x["time_idx"] = x.groupby(group_col).cumcount()

    # calendar features
    x["hour"] = x["ts_utc"].dt.hour
    x["day"] = x["ts_utc"].dt.day
    x["weekday"] = x["ts_utc"].dt.weekday
    x["month"] = x["ts_utc"].dt.month

    return x[["ts_utc", target_col, group_col, "time_idx", "hour", "day", "weekday", "month"]]
