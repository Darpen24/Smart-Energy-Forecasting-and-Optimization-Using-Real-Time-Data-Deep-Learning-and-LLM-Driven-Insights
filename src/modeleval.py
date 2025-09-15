# modeleval.py — compute daily KPIs over actual vs predicted
import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from db import get_connection

def mape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / np.maximum(1e-6, y))) * 100)

def main():
    with get_connection() as conn:
        df = pd.read_sql_query(text("""
            SELECT a.ts_utc, a.load AS actual, p.predicted_load AS pred
            FROM raw_data a
            JOIN predicted_load p ON a.ts_utc = p.ts_utc
            ORDER BY a.ts_utc
        """), con=conn)

    if df.empty:
        print(" No overlap between actuals and predictions. Skipping eval.")
        return

    rmse = mean_squared_error(df["actual"], df["pred"], squared=False)
    mae  = mean_absolute_error(df["actual"], df["pred"])
    mape_val = mape(df["actual"], df["pred"])
    r2   = r2_score(df["actual"], df["pred"])

    print(f" Eval — RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape_val:.2f}%, R2: {r2:.3f}")

    # write KPIs to DB (optional)
    try:
        with get_connection() as conn:
            (pd.DataFrame([{"rmse": rmse, "mae": mae, "mape": mape_val, "r2": r2}])
             .assign(run_ts=pd.Timestamp.utcnow())
             .to_sql("kpi_daily", con=conn, index=False, if_exists="append"))
        print(" KPIs appended to table 'kpi_daily'")
    except Exception as e:
        print(f" KPI write skipped: {e}")

if __name__ == "__main__":
    main()
