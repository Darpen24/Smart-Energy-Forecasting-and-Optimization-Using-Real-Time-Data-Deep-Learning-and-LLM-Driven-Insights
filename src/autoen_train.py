# autoen_train.py
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy import text
from sklearn.preprocessing import MinMaxScaler

from db import get_connection

# -----------------------------
# 1) Load raw data
# -----------------------------
with get_connection() as conn:
    df = pd.read_sql_query(text("SELECT ts_utc, load FROM raw_data ORDER BY ts_utc"), con=conn)

df = df.dropna(subset=["load"]).reset_index(drop=True)

# -----------------------------
# 2) Scale values
# -----------------------------
scaler = MinMaxScaler()
df["load_scaled"] = scaler.fit_transform(df[["load"]])

load_scaled = df["load_scaled"].values.astype("float32")

# -----------------------------
# 3) Build 24h windows
# -----------------------------
window = 24

def make_windows(arr, W):
    N = len(arr)
    return np.stack([arr[i:i+W] for i in range(0, N - W + 1)], axis=0)

X = make_windows(load_scaled, window)  # shape [N, 24]
X_t = torch.tensor(X, dtype=torch.float32)

# -----------------------------
# 4) Define Autoencoder
# -----------------------------
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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 5) Train
# -----------------------------
epochs = 20  # can reduce for quick dev
for ep in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_t)
    loss = criterion(out, X_t)
    loss.backward()
    optimizer.step()
    if ep % 5 == 0:
        print(f"Epoch {ep}, Loss: {loss.item():.6f}")

# -----------------------------
# 6) Compute threshold
# -----------------------------
model.eval()
with torch.no_grad():
    recon = model(X_t).numpy()
train_err = ((X - recon) ** 2).mean(axis=1)
threshold = float(train_err.mean() + 3 * train_err.std())

# -----------------------------
# 7) Save artifacts
# -----------------------------
torch.save(model.state_dict(), "autoencoder_model.pt")
with open("autoencoder_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("autoencoder_threshold.json", "w") as f:
    json.dump({"window": window, "threshold": threshold}, f)

print(" Autoencoder training complete")
print(f"   Model: autoencoder_model.pt")
print(f"   Scaler: autoencoder_scaler.pkl")
print(f"   Threshold: autoencoder_threshold.json (thr={threshold:.6f})")
