import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.preprocess import preprocess_dataframe, get_data_summary
from src.train_model import train_model
from src.predict import make_predictions, get_feature_importance

st.set_page_config(page_title="EV Charging Demand Prediction", page_icon="⚡", layout="wide")


plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#0E1117",
    "figure.facecolor": "#0E1117",
    "grid.color": "#1E2130",
    "text.color": "#FAFAFA",
    "axes.labelcolor": "#FAFAFA",
    "xtick.color": "#FAFAFA",
    "ytick.color": "#FAFAFA",
})

st.title("EV Charging Demand Prediction System")
st.caption("Upload charging data · Pick a model · Get forecasts")

uploaded_file = st.file_uploader("Upload EV Charging Volume CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)
df_long = preprocess_dataframe(df)
summary = get_data_summary(df_long)

st.subheader("Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Traffic Zones", summary['num_zones'])
c2.metric("Total Records", f"{summary['total_records']:,}")
c3.metric("Avg Volume (kWh)", summary['avg_volume'])
c4.metric("Peak Volume (kWh)", f"{summary['peak_volume']:,.0f}")

with st.expander("View raw data"):
    st.dataframe(df_long.head(200), width="stretch")

# demand patterns 
st.divider()
st.subheader("Demand Patterns")

col_hourly, col_weekly = st.columns(2)

with col_hourly:
    st.markdown("**Hourly Demand**")
    hourly = df_long.groupby('hour')['volume_kwh'].mean()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hourly.index, hourly.values, color='#4FC3F7', linewidth=2, marker='o', markersize=3)
    ax.fill_between(hourly.index, hourly.values, alpha=0.1, color='#4FC3F7')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg kWh")
    ax.set_xticks(range(0, 24, 3))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col_weekly:
    st.markdown("**Weekday vs Weekend**")
    df_long['day_type'] = df_long['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    day_hour = df_long.groupby(['day_type', 'hour'])['volume_kwh'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for dtype, color in [('Weekday', '#4FC3F7'), ('Weekend', '#FF8A65')]:
        subset = day_hour[day_hour['day_type'] == dtype]
        ax.plot(subset['hour'], subset['volume_kwh'], color=color, linewidth=2, label=dtype)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg kWh")
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# monthly + top zones 
st.divider()
col_month, col_zones = st.columns(2)

with col_month:
    st.markdown("**Monthly Average Demand**")
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly = df_long.groupby('month')['volume_kwh'].mean()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(
        [month_names[m] for m in monthly.index],
        monthly.values,
        color='#4FC3F7',
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg kWh")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col_zones:
    st.markdown("**Top 10 Zones by Demand**")
    top_zones = (df_long.groupby('TAZID')['volume_kwh'].sum()
                 .nlargest(10).sort_values())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(
        top_zones.index.astype(str),
        top_zones.values,
        color='#FF8A65',
    )
    ax.set_xlabel("Total kWh")
    ax.set_ylabel("Zone")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# model config & run 
st.divider()
col_model, col_btn = st.columns([2, 1])
with col_model:
    model_type = st.selectbox("Model", ["Linear Regression", "Random Forest"])
with col_btn:
    st.write("")
    run = st.button("Run Prediction", type="primary", use_container_width=True)

if not run:
    st.stop()

with st.spinner("Training..."):
    model, X_train, X_test, y_train, y_test, feature_names = train_model(df_long, model_type)
    predictions, metrics = make_predictions(model, X_test, y_test)
    importances = get_feature_importance(model, feature_names)

# metrics
st.divider()
st.subheader("Model Performance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("R² Score", metrics['r2'])
m2.metric("MAE", metrics['mae'])
m3.metric("RMSE", metrics['rmse'])
m4.metric("MAPE", f"{metrics['mape']}%")

# actual vs predicted 
st.divider()
st.subheader("Actual vs Predicted")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y_test.values[:200], color='#4FC3F7', linewidth=1.5, label='Actual')
ax.plot(predictions[:200], color='#FF8A65', linewidth=1.5, linestyle='--', label='Predicted')
ax.set_xlabel("Time Index")
ax.set_ylabel("Volume (kWh)")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# feature importance 
st.divider()
st.subheader("Feature Importance")
if importances:
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(sorted_imp.keys()), list(sorted_imp.values()), color='#4FC3F7')
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.success("Done!")