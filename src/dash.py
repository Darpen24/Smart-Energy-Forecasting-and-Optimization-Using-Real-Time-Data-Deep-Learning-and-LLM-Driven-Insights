import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import text
from db import get_engine

# --- CONFIG ---
st.set_page_config(page_title="Smart Energy Dashboard", layout="wide")
st.title("⚡ Smart Energy Dashboard")

engine = get_engine()

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
with engine.connect() as conn:
    df_actual = pd.read_sql(text("SELECT ts_utc, load FROM raw_data ORDER BY ts_utc"), conn)
    df_pred = pd.read_sql(text("SELECT ts_utc, predicted_load FROM predicted_load ORDER BY ts_utc"), conn)
    try:
        df_solar_pred = pd.read_sql(text("SELECT ts_utc, predicted_solar FROM predicted_solar ORDER BY ts_utc"), conn)
    except:
        df_solar_pred = pd.DataFrame(columns=["ts_utc", "predicted_solar"])
    try:
        df_solar_actual = pd.read_sql(text("SELECT ts_utc, solar_generation FROM solar_clean ORDER BY ts_utc"), conn)
    except:
        df_solar_actual = pd.read_sql(text("SELECT ts_utc, solar_generation FROM solar_data ORDER BY ts_utc"), conn)
    try:
        df_app_pred = pd.read_sql(text("SELECT ts_utc, appliance, predicted_usage_kwh FROM predicted_appliance ORDER BY ts_utc"), conn)
    except:
        df_app_pred = pd.DataFrame(columns=["ts_utc", "appliance", "predicted_usage_kwh"])

# Fix timestamps
for _df in (df_actual, df_pred, df_solar_pred, df_solar_actual, df_app_pred):
    if not _df.empty:
        _df["ts_utc"] = pd.to_datetime(_df["ts_utc"], utc=True).dt.tz_convert(None)

# Merge actual + predicted + solar
df = (
    df_actual
    .merge(df_pred, on="ts_utc", how="outer")
    .merge(df_solar_pred, on="ts_utc", how="outer")
    .merge(df_solar_actual.rename(columns={"solar_generation": "actual_solar"}), on="ts_utc", how="left")
    .sort_values("ts_utc")
)
df["solar_used"] = df["predicted_solar"].fillna(df["actual_solar"])

# Appliance loader
def load_appliance_long(_engine):
    candidate_tables = ["appliance_usage", "appliance_usage_stage", "dataset_stage", "appliance_stage"]
    with _engine.connect() as conn:
        for t in candidate_tables:
            try:
                df_wide = pd.read_sql(text(f"SELECT * FROM {t} ORDER BY 1"), conn)
                if df_wide.empty:
                    continue
                ts_col = df_wide.columns[0]
                numeric_cols = df_wide.select_dtypes(include=[np.number]).columns.tolist()
                value_cols = [c for c in df_wide.columns if c != ts_col and c in numeric_cols]
                df_long = df_wide.melt(id_vars=[ts_col], value_vars=value_cols,
                                       var_name="appliance", value_name="usage_kwh")
                df_long = df_long.rename(columns={ts_col: "ts_utc"})
                df_long["ts_utc"] = pd.to_datetime(df_long["ts_utc"], utc=True).dt.tz_convert(None)
                return df_long.dropna(subset=["usage_kwh"])
            except Exception:
                continue
    return pd.DataFrame(columns=["ts_utc", "appliance", "usage_kwh"])

df_app_actual = load_appliance_long(engine)
if not df_app_actual.empty or not df_app_pred.empty:
    df_app_combined = pd.concat([
        df_app_actual.rename(columns={"usage_kwh": "value"}).assign(source="actual"),
        df_app_pred.rename(columns={"predicted_usage_kwh": "value"}).assign(source="predicted")
    ], ignore_index=True)
else:
    df_app_combined = pd.DataFrame(columns=["ts_utc", "appliance", "value", "source"])

# ---------------------------------------------------------------------
# FILTERING
# ---------------------------------------------------------------------
if df["ts_utc"].notna().any():
    min_dt = df["ts_utc"].min()
    max_dt = df["ts_utc"].max()

    st.subheader("🗓️ Filter by Date")
    date_range = st.date_input(
        "Select date range (calendar)",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date()
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = min_dt, max_dt

    mask_range = (df["ts_utc"] >= start_date) & (df["ts_utc"] <= end_date)
    df_range = df.loc[mask_range].copy()
else:
    st.error("No data available to display. Showing last 30 days from today.")
    today = pd.Timestamp.today()
    last_30 = pd.date_range(today - pd.Timedelta(days=30), today, freq="H")
    df_range = pd.DataFrame({"ts_utc": last_30, "predicted_load": np.nan, "load": np.nan})

# ---------------------------------------------------------------------
# KPIs (converted to GWh)
# ---------------------------------------------------------------------
total_usage_gwh = df_range["predicted_load"].fillna(df_range["load"]).sum() / 1_000_000
solar_used = df_range["solar_used"].sum() if "solar_used" in df_range.columns else 0
solar_pct = (solar_used / (total_usage_gwh * 1_000_000) * 100) if total_usage_gwh > 0 else 0
peak_load_gwh = (df_range["predicted_load"].fillna(df_range["load"]).max() or 0) / 1_000_000
daily_usage_gwh = df_range.set_index("ts_utc")["predicted_load"].fillna(df_range["load"]).resample("D").sum().mean() / 1_000_000

st.subheader("📊 Insights")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Usage", f"{total_usage_gwh:,.2f} GWh")
k2.metric("Solar Contribution", f"{solar_pct:.1f}%")
k3.metric("Peak Load (Max Hourly)", f"{peak_load_gwh:.3f} GWh")
k4.metric("Avg Daily Usage", f"{daily_usage_gwh:.2f} GWh/day" if not pd.isna(daily_usage_gwh) else "N/A")

# Appliance-specific KPI
if appliance_selected := (
    st.selectbox("🔌 Filter by Appliance", ["(All)"] + sorted(df_app_combined["appliance"].unique()))
    if not df_app_combined.empty else None
):
    if appliance_selected != "(All)":
        app_range = df_app_combined[(df_app_combined["ts_utc"] >= start_date) & (df_app_combined["ts_utc"] <= end_date)]
        app_range = app_range[app_range["appliance"] == appliance_selected]
        kwh_app = app_range["value"].sum()
        st.metric(f"{appliance_selected} Usage", f"{kwh_app/1_000_000:.2f} GWh")

# ---------------------------------------------------------------------
# CHARTS (better layout)
# ---------------------------------------------------------------------
st.subheader("📈 Visual Insights")
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Predicted vs Actual Load (GWh)")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(df_range["ts_utc"], df_range["load"]/1_000_000, label="Actual Load", alpha=0.6)
    ax1.plot(df_range["ts_utc"], df_range["predicted_load"]/1_000_000, label="Predicted Load", alpha=0.8)
    ax1.set_ylabel("Load (GWh)")
    ax1.set_xlabel("Date")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    st.pyplot(fig1)

with c2:
    if "solar_used" in df_range.columns:
        st.markdown("### Daily Load vs Solar %")
        daily = df_range.set_index("ts_utc").resample("D").sum().reset_index()
        daily["solar_pct"] = (daily["solar_used"] / daily["predicted_load"].fillna(daily["load"])) * 100
        fig_combo, ax1 = plt.subplots(figsize=(6, 3))
        ax2 = ax1.twinx()
        ax1.bar(daily["ts_utc"], daily["predicted_load"]/1_000_000, label="Daily Load (GWh)", color="skyblue")
        ax2.plot(daily["ts_utc"], daily["solar_pct"], color="green", marker="o", label="Solar %")
        ax1.set_ylabel("Daily Load (GWh)")
        ax2.set_ylabel("Solar %")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax1.grid(alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
        st.pyplot(fig_combo)

c3, c4 = st.columns(2)
with c3:
    st.markdown("### Load Distribution Histogram")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.hist(df_range["predicted_load"].fillna(df_range["load"]) / 1_000_000, bins=30,
             color="steelblue", edgecolor="black")
    ax3.set_xlabel("Load (GWh)")
    ax3.set_ylabel("Frequency")
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)

with c4:
    if not df_app_combined.empty:
        app_range = df_app_combined[(df_app_combined["ts_utc"] >= start_date) & (df_app_combined["ts_utc"] <= end_date)]
        if not app_range.empty:
            st.markdown("### Appliance Contribution (%)")
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
            values = app_range.groupby("appliance")["value"].sum()
            wedges, _, autotexts = ax_pie.pie(
                values,
                autopct='%1.1f%%',
                startangle=90,
                pctdistance=1.2
            )
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.6))
            ax_pie.legend(wedges, values.index, title="Appliances",
                          loc="center left", bbox_to_anchor=(1, 0.5))
            ax_pie.set_title("Share of Energy by Appliance")
            st.pyplot(fig_pie)

st.markdown("### Hourly Load Heatmap")
df_heat = df_range.copy()
df_heat["date"] = df_heat["ts_utc"].dt.date
df_heat["hour"] = df_heat["ts_utc"].dt.hour
heatmap_data = df_heat.pivot_table(index="hour", columns="date", values="predicted_load", aggfunc="mean")/1_000_000
fig_hm, ax_hm = plt.subplots(figsize=(10, 3))
im = ax_hm.imshow(heatmap_data, aspect="auto", cmap="viridis")
ax_hm.set_xlabel("Date")
ax_hm.set_ylabel("Hour of Day")
ax_hm.set_title("Hourly Load Pattern (GWh)")
plt.colorbar(im, ax=ax_hm, label="GWh")
st.pyplot(fig_hm)

# ---------------------------------------------------------------------
# CHAT ASSISTANT (Dropdown-Based)
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("💬 Smart Energy Chat Assistant")

query_options = [
    "Select a question...",
    "🔌 Total Usage in Selected Range",
    "☀️ Solar Contribution Percentage",
    "📈 Peak Load & Timestamp",
    "🏠 Most Used Appliance",
    "📊 Predicted vs Actual Load Summary"
]

user_choice = st.selectbox("Ask a question:", query_options)

if user_choice != "Select a question...":
    with st.chat_message("assistant"):
        if user_choice == "🔌 Total Usage in Selected Range":
            total_kwh = df_range["predicted_load"].fillna(df_range["load"]).sum()
            st.markdown(f"**🔌 Total Usage:** {total_kwh/1_000_000:.2f} GWh during the selected period.")
        elif user_choice == "☀️ Solar Contribution Percentage":
            solar_used = df_range["solar_used"].sum() if "solar_used" in df_range.columns else 0
            solar_pct = (solar_used / df_range["predicted_load"].fillna(df_range["load"]).sum()) * 100
            st.markdown(f"**☀️ Solar Contribution:** {solar_pct:.1f}% of total energy consumption came from solar.")
        elif user_choice == "📈 Peak Load & Timestamp":
            peak_val = df_range["predicted_load"].fillna(df_range["load"]).max()
            peak_time = df_range.loc[df_range["predicted_load"].fillna(df_range["load"]).idxmax(), "ts_utc"]
            st.markdown(f"**📈 Peak Load:** {peak_val/1_000_000:.3f} GWh at **{peak_time.strftime('%Y-%m-%d %H:%M')}**.")
        elif user_choice == "🏠 Most Used Appliance":
            if not df_app_combined.empty:
                top_app = (
                    df_app_combined[(df_app_combined["ts_utc"] >= start_date) & (df_app_combined["ts_utc"] <= end_date)]
                    .groupby("appliance")["value"].sum().sort_values(ascending=False)
                )
                if not top_app.empty:
                    st.markdown(f"**🏠 Top Appliance:** {top_app.index[0]} with {top_app.iloc[0]/1_000_000:.2f} GWh usage.")
                else:
                    st.markdown("⚠️ No appliance data available for this range.")
            else:
                st.markdown("⚠️ No appliance data found in database.")
        elif user_choice == "📊 Predicted vs Actual Load Summary":
            actual = df_range["load"].dropna().mean()
            predicted = df_range["predicted_load"].dropna().mean()
            diff = (predicted - actual) / actual * 100 if actual > 0 else 0
            st.markdown(
                f"**📊 Summary:** Average predicted load = {predicted/1_000:.2f} MWh, "
                f"actual load = {actual/1_000:.2f} MWh. Difference: {diff:.2f}%"
            )

st.markdown("---")
