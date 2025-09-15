# tft_predict.py — Final Wide-Table Compatible Version ✅ (Output Size Fixed + Row Logs)

import os, json, numpy as np, pandas as pd, torch
from sqlalchemy import text
from pytorch_lightning import seed_everything
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE

from db import get_engine
from tft_utils import prepare_time_series_df

MODEL_PATH = "tft_model_fast.pt"  # must match training save
SERIES_META_PATH = "tft_series_meta.json"
VERBOSE = True

def log(*args):
    if VERBOSE:
        print(*args, flush=True)

def _utc(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert(None)

def ensure_pred_tables(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS predicted_load (
      ts_utc TIMESTAMP PRIMARY KEY,
      predicted_load DOUBLE PRECISION
    );
    CREATE TABLE IF NOT EXISTS predicted_solar (
      ts_utc TIMESTAMP PRIMARY KEY,
      predicted_solar DOUBLE PRECISION
    );
    CREATE TABLE IF NOT EXISTS predicted_appliance (
      ts_utc TIMESTAMP,
      appliance TEXT,
      predicted_usage_kwh DOUBLE PRECISION,
      PRIMARY KEY (ts_utc, appliance)
    );
    CREATE TABLE IF NOT EXISTS predicted_load_2min (
      ts_utc TIMESTAMP PRIMARY KEY,
      predicted_load DOUBLE PRECISION
    );
    CREATE TABLE IF NOT EXISTS predicted_solar_2min (
      ts_utc TIMESTAMP PRIMARY KEY,
      predicted_solar DOUBLE PRECISION
    );
    CREATE TABLE IF NOT EXISTS predicted_appliance_2min (
      ts_utc TIMESTAMP,
      appliance TEXT,
      predicted_usage_kwh DOUBLE PRECISION,
      PRIMARY KEY (ts_utc, appliance)
    );
    """
    with engine.begin() as conn:
        for stmt in ddl.strip().split(";\n"):
            if stmt.strip():
                conn.execute(text(stmt))

def _combined_actuals_plus_preds(engine, series_name):
    with engine.connect() as conn:
        if series_name == "grid_load":
            df_a = pd.read_sql(text("SELECT ts_utc, load FROM appliance_usage ORDER BY ts_utc"), conn)
            try:
                df_p = pd.read_sql(text("SELECT ts_utc, predicted_load FROM predicted_load ORDER BY ts_utc"), conn)
            except:
                df_p = pd.DataFrame(columns=["ts_utc", "predicted_load"])
            if not df_a.empty: df_a["ts_utc"] = _utc(df_a["ts_utc"])
            if not df_p.empty: df_p["ts_utc"] = _utc(df_p["ts_utc"])
            df = pd.merge(df_a, df_p, on="ts_utc", how="outer")
            df["value"] = np.where(df["load"].notna(), df["load"], df["predicted_load"])
            return df[["ts_utc", "value"]].dropna().assign(series="grid_load")

        if series_name == "solar":
            try:
                df_a = pd.read_sql(text("SELECT ts_utc, solar_generation FROM solar_clean ORDER BY ts_utc"), conn)
            except:
                df_a = pd.read_sql(text("SELECT ts_utc, solar_generation FROM solar_data ORDER BY ts_utc"), conn)
            try:
                df_p = pd.read_sql(text("SELECT ts_utc, predicted_solar FROM predicted_solar ORDER BY ts_utc"), conn)
            except:
                df_p = pd.DataFrame(columns=["ts_utc", "predicted_solar"])
            if not df_a.empty: df_a["ts_utc"] = _utc(df_a["ts_utc"])
            if not df_p.empty: df_p["ts_utc"] = _utc(df_p["ts_utc"])
            df = pd.merge(df_a, df_p, on="ts_utc", how="outer")
            df["value"] = np.where(df["solar_generation"].notna(), df["solar_generation"], df["predicted_solar"])
            return df[["ts_utc", "value"]].dropna().assign(series="solar")

        if series_name.startswith("appliance:"):
            col = series_name.split("appliance:", 1)[1]
            try:
                df_a = pd.read_sql(text(f"SELECT ts_utc, {col} AS usage_kwh FROM appliance_usage ORDER BY ts_utc"), conn)
            except Exception as e:
                log(f"⚠️ Column {col} not found in appliance_usage: {e}")
                return pd.DataFrame(columns=["ts_utc", "value", "series"])
            try:
                df_p = pd.read_sql(
                    text("SELECT ts_utc, predicted_usage_kwh FROM predicted_appliance WHERE appliance=:a ORDER BY ts_utc"),
                    conn, params={"a": col}
                )
            except:
                df_p = pd.DataFrame(columns=["ts_utc", "predicted_usage_kwh"])
            if not df_a.empty: df_a["ts_utc"] = _utc(df_a["ts_utc"])
            if not df_p.empty: df_p["ts_utc"] = _utc(df_p["ts_utc"])
            df = pd.merge(df_a, df_p, on="ts_utc", how="outer")
            df["value"] = np.where(df["usage_kwh"].notna(), df["usage_kwh"], df["predicted_usage_kwh"])
            return df[["ts_utc", "value"]].dropna().assign(series=series_name)

    return pd.DataFrame(columns=["ts_utc", "value", "series"])

def _resample_to_2min(df):
    if df.empty:
        return df
    s = df.set_index("ts_utc")["value"].astype(float).sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="2T")
    s2 = s.reindex(idx).interpolate("time").clip(lower=0)
    return s2.reset_index().rename(columns={"index": "ts_utc", "value": "value"})

def main():
    seed_everything(42)
    engine = get_engine()
    ensure_pred_tables(engine)

    with open(SERIES_META_PATH, "r") as f:
        meta = json.load(f)
    series = meta["series"]
    enc_len = meta["max_encoder_length"]
    pred_len = meta["max_prediction_length"]

    latest_list, future_list, series_in_run, future_index = [], [], [], {}

    for s in series:
        base = _combined_actuals_plus_preds(engine, s)
        log(f"Series {s}: {len(base)} rows found")
        if base.empty:
            continue
        latest = base.tail(enc_len)
        last_ts = latest["ts_utc"].max()
        future_ts = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=pred_len, freq="H")
        latest_list.append(latest.assign(series=s))
        future_list.append(pd.DataFrame({"ts_utc": future_ts, "value": 0.0, "series": s}))
        series_in_run.append(s)
        future_index[s] = future_ts

    if not latest_list:
        log("No data found for any series — nothing to predict.")
        return

    df_predict = pd.concat(latest_list + future_list)
    df_predict = prepare_time_series_df(df_predict, "value", "series")

    predict_ds = TimeSeriesDataSet(
        df_predict,
        time_idx="time_idx",
        target="value",
        group_ids=["series"],
        max_encoder_length=enc_len,
        max_prediction_length=pred_len,
        time_varying_known_reals=["time_idx", "hour", "day", "weekday", "month"],
        time_varying_unknown_reals=["value"],
        predict_mode=True,
    )
    loader = predict_ds.to_dataloader(train=False, batch_size=max(1, len(series_in_run)), num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("Step 4: loading model …")

    # ✅ FIX: Force output_size=1 to match training
    model = TemporalFusionTransformer.from_dataset(
        predict_ds,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        loss=SMAPE(),
        output_size=1
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    log(f"Loaded model state_dict. Missing: {missing}, Unexpected: {unexpected}")
    model.to(device).eval()

    with torch.no_grad():
        preds = model.predict(loader)
    preds = np.atleast_2d(preds.cpu().numpy())

    rows = []
    for i, s in enumerate(series_in_run):
        vals = preds[i][: len(future_index[s])]
        rows.append(pd.DataFrame({"ts_utc": future_index[s], "value": vals, "series": s}))
    pred_hourly = pd.concat(rows)

    pred_2min = pd.concat([_resample_to_2min(g).assign(series=name)
                           for name, g in pred_hourly.groupby("series")])

    # ---- Write predictions to DB ----
    with engine.begin() as conn:
        # Grid load
        grid = pred_hourly[pred_hourly.series == "grid_load"].copy()
        if not grid.empty:
            conn.execute(text("DELETE FROM predicted_load WHERE ts_utc=ANY(:ts)"),
                         {"ts": grid.ts_utc.dt.to_pydatetime().tolist()})
            grid.rename(columns={"value": "predicted_load"}, inplace=True)
            grid[["ts_utc", "predicted_load"]].to_sql("predicted_load", con=conn, if_exists="append", index=False)
            log(f"Inserted {len(grid)} rows → predicted_load")

        # Solar
        solar = pred_hourly[pred_hourly.series == "solar"].copy()
        if not solar.empty:
            conn.execute(text("DELETE FROM predicted_solar WHERE ts_utc=ANY(:ts)"),
                         {"ts": solar.ts_utc.dt.to_pydatetime().tolist()})
            solar.rename(columns={"value": "predicted_solar"}, inplace=True)
            solar[["ts_utc", "predicted_solar"]].to_sql("predicted_solar", con=conn, if_exists="append", index=False)
            log(f"Inserted {len(solar)} rows → predicted_solar")

        # Appliances
        apps = pred_hourly[pred_hourly.series.str.startswith("appliance:")].copy()
        if not apps.empty:
            apps["appliance"] = apps.series.str.split("appliance:", 1).str[1]
            out = apps[["ts_utc", "appliance", "value"]].rename(columns={"value": "predicted_usage_kwh"})
            conn.execute(text("DELETE FROM predicted_appliance WHERE ts_utc=ANY(:ts)"),
                         {"ts": out.ts_utc.dt.to_pydatetime().tolist()})
            out.to_sql("predicted_appliance", con=conn, if_exists="append", index=False)
            log(f"Inserted {len(out)} rows → predicted_appliance")

        # 2-min tables
        grid2 = pred_2min[pred_2min.series == "grid_load"].copy()
        if not grid2.empty:
            conn.execute(text("DELETE FROM predicted_load_2min WHERE ts_utc=ANY(:ts)"),
                         {"ts": grid2.ts_utc.dt.to_pydatetime().tolist()})
            grid2.rename(columns={"value": "predicted_load"}, inplace=True)
            grid2[["ts_utc", "predicted_load"]].to_sql("predicted_load_2min", con=conn, if_exists="append", index=False)
            log(f"Inserted {len(grid2)} rows → predicted_load_2min")

        solar2 = pred_2min[pred_2min.series == "solar"].copy()
        if not solar2.empty:
            conn.execute(text("DELETE FROM predicted_solar_2min WHERE ts_utc=ANY(:ts)"),
                         {"ts": solar2.ts_utc.dt.to_pydatetime().tolist()})
            solar2.rename(columns={"value": "predicted_solar"}, inplace=True)
            solar2[["ts_utc", "predicted_solar"]].to_sql("predicted_solar_2min", con=conn, if_exists="append", index=False)
            log(f"Inserted {len(solar2)} rows → predicted_solar_2min")

        apps2 = pred_2min[pred_2min.series.str.startswith("appliance:")].copy()
        if not apps2.empty:
            apps2["appliance"] = apps2.series.str.split("appliance:", 1).str[1]
            out2 = apps2[["ts_utc", "appliance", "value"]].rename(columns={"value": "predicted_usage_kwh"})
            conn.execute(text("DELETE FROM predicted_appliance_2min WHERE ts_utc=ANY(:ts)"),
                         {"ts": out2.ts_utc.dt.to_pydatetime().tolist()})
            out2.to_sql("predicted_appliance_2min", con=conn, if_exists="append", index=False)
            log(f"Inserted {len(out2)} rows → predicted_appliance_2min")

    print("✅ Predictions stored successfully.")

if __name__ == "__main__":
    main()
