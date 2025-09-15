# tft_train_fast.py — Clean & Fast TFT Training ✅ (Wide Table Compatible)

import json
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import seed_everything, Trainer
from sqlalchemy import text

from db import get_engine
from tft_utils import make_series, prepare_time_series_df

# -----------------------------
# SETTINGS
# -----------------------------
MAX_ENCODER_LENGTH = 168   # past 7 days
MAX_PRED_LENGTH = 24       # predict next 24h
MODEL_PATH = "tft_model_fast.pt"
SERIES_META_PATH = "tft_series_meta.json"

# -----------------------------
# LOAD DATA FROM DATABASE
# -----------------------------
engine = get_engine()
with engine.connect() as conn:
    # Load grid load (from appliance_usage table load column)
    df_load = pd.read_sql(text("SELECT ts_utc, load FROM appliance_usage ORDER BY ts_utc"), conn)

    # Load solar
    try:
        df_solar = pd.read_sql(text("SELECT ts_utc, solar_generation FROM solar_clean ORDER BY ts_utc"), conn)
    except:
        try:
            df_solar = pd.read_sql(text("SELECT ts_utc, solar_generation FROM solar_data ORDER BY ts_utc"), conn)
        except:
            df_solar = pd.DataFrame(columns=["ts_utc", "solar_generation"])

    # Load appliances from wide table
    wide = pd.read_sql(text("SELECT * FROM appliance_usage ORDER BY ts_utc"), conn)
    appliance_cols = ["heating", "water_heating", "appliances_lighting", "cooking", "cooling", "other"]

    # Melt to long format: ts_utc, appliance, usage_kwh
    df_app_long = wide.melt(id_vars=["ts_utc"], value_vars=appliance_cols,
                            var_name="appliance", value_name="usage_kwh")
    df_app_long["ts_utc"] = pd.to_datetime(df_app_long["ts_utc"], utc=True).dt.tz_convert(None)

# -----------------------------
# COMBINE DATA INTO ONE LONG DF
# -----------------------------
df_all_list = []

# grid load
if not df_load.empty:
    df_all_list.append(make_series(df_load, "ts_utc", "load", "grid_load"))

# solar
if not df_solar.empty:
    df_all_list.append(make_series(df_solar, "ts_utc", "solar_generation", "solar"))

# appliances
appliance_names = []
if not df_app_long.empty:
    appliance_names = sorted(df_app_long["appliance"].unique().tolist())
    for a in appliance_names:
        dfa = df_app_long[df_app_long["appliance"] == a][["ts_utc", "usage_kwh"]].copy()
        df_all_list.append(make_series(dfa, "ts_utc", "usage_kwh", f"appliance:{a}"))

if len(df_all_list) == 0:
    raise SystemExit("No data available for training.")

df_all = pd.concat(df_all_list, ignore_index=True)
df_all = prepare_time_series_df(df_all, target_col="value", group_col="series")

# -----------------------------
# DATASET & DATALOADERS
# -----------------------------
training = TimeSeriesDataSet(
    df_all,
    time_idx="time_idx",
    target="value",
    group_ids=["series"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PRED_LENGTH,
    time_varying_known_reals=["time_idx", "hour", "day", "weekday", "month"],
    time_varying_unknown_reals=["value"],
    target_normalizer=GroupNormalizer(groups=["series"]),
)

train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=4)
val_loader   = training.to_dataloader(train=False, batch_size=64, num_workers=4)

# -----------------------------
# MODEL & TRAINING
# -----------------------------
seed_everything(42)

model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=SMAPE(),
    log_interval=10,
    log_val_interval=1,
)

logger = TensorBoardLogger("lightning_logs", name="tft_fast")
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

trainer = Trainer(
    max_epochs=1,  # quick training
    gradient_clip_val=0.1,
    logger=logger,
    callbacks=[checkpoint_callback],
    enable_checkpointing=True
)

# -----------------------------
# RUN TRAINING
# -----------------------------
if __name__ == "__main__":
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    torch.save(model.state_dict(), MODEL_PATH)
    meta = {
        "series": sorted(df_all["series"].unique().tolist()),
        "appliances": appliance_names,
        "max_encoder_length": MAX_ENCODER_LENGTH,
        "max_prediction_length": MAX_PRED_LENGTH,
    }
    with open(SERIES_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Model trained for 2 epochs and saved to {MODEL_PATH}")
    print(f"✅ Series meta saved to {SERIES_META_PATH}")
    print("✅ Best checkpoint saved → checkpoints/best-checkpoint.ckpt")
