# autoen_detect.py
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy import text
from db import get_connection

# -----------------------------
# 1️⃣ Load trained artifacts
# -----------------------------
with open("autoencoder_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("autoencoder_threshold.json", "r") as f:
    cfg = json.load(f)
window = int(cfg["window"])
threshold = float(cfg["threshold"])

class Autoencoder(nn.Module):
    def __init__(self, in_dim=24, hidden=64, z_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(in_dim=window, hidden=64, z_dim=16)
model.load_state_dict(torch.load("autoencoder_model.pt", map_location="cpu"))
model.eval()

# -----------------------------
# 2️⃣ Load recent data (predictions)
# -----------------------------
with get_connection() as conn:
    df = pd.read_sql_query(
        text("SELECT ts_utc, predicted_load AS load FROM predicted_load ORDER BY ts_utc"),
        con=conn
    )

df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
df = df.dropna(subset=["load"]).reset_index(drop=True)

# -----------------------------
# 3️⃣ Scale + make 24h windows
# -----------------------------
scaled = scaler.transform(df[["load"]]).astype("float32").flatten()

def make_windows(arr, W):
    N = len(arr)
    if N < W:
        return np.empty((0, W)), []
    X, end_ts = [], []
    for i in range(N - W + 1):
        X.append(arr[i:i+W])
        end_ts.append(df["ts_utc"].iloc[i+W-1])
    return np.stack(X, axis=0), end_ts

X, end_timestamps = make_windows(scaled, window)
if X.shape[0] == 0:
    print(" Not enough data for anomaly detection.")
    raise SystemExit(0)

X_t = torch.tensor(X, dtype=torch.float32)

# -----------------------------
# 4️⃣ Compute reconstruction error
# -----------------------------
with torch.no_grad():
    recon = model(X_t).numpy()
errors = ((X - recon) ** 2).mean(axis=1)

# -----------------------------
# 5️⃣ Flag anomalies using saved threshold
# -----------------------------
flags = errors > threshold
anoms = pd.DataFrame({
    "ts_utc": end_timestamps,
    "reconstruction_error": errors,
    "is_anomaly": flags.astype(int)
})

print(f" Detected {anoms['is_anomaly'].sum()} anomalies out of {len(anoms)} windows")

# -----------------------------
# 6️ Save anomalies to DB
# -----------------------------
with get_connection() as conn:
    anoms.to_sql("anomalies_detected", con=conn, index=False, if_exists="replace")

print(" Results saved to table 'anomalies_detected'")
