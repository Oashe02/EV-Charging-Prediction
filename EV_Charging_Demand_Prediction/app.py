import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.preprocess import preprocess_dataframe, get_data_summary
from src.train_model import train_model
from src.predict import make_predictions, get_feature_importance
from src.agent import run_agentic_workflow
from src.pdf_generator import generate_pdf_report

st.set_page_config(page_title="EV Charging Project", layout="wide")

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

# Sidebar Configuration
st.sidebar.markdown("## Configuration")
api_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="Enter key for agent report...")
st.sidebar.caption("Provide an API key to generate the AI planning report.")

st.title("EV Charging Demand and Infrastructure Planning System")
st.caption("College Project: Milestone 1 and 2")

uploaded_file = st.file_uploader("Upload EV Charging Data (CSV)", type=["csv"])

if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df_long = preprocess_dataframe(df)
    summary = get_data_summary(df_long)
    return df_long, summary

df_long, summary = load_data(uploaded_file)

tab_dash, tab_ai, tab_scenario = st.tabs(["Dashboard", "Infrastructure Planner", "Scenario Analysis"])

with tab_dash:
    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traffic Zones", summary['num_zones'])
    c2.metric("Total Records", f"{summary['total_records']:,}")
    c3.metric("Avg Volume (kWh)", summary['avg_volume'])
    c4.metric("Peak Volume (kWh)", f"{summary['peak_volume']:,.0f}")

    with st.expander("View Data Table"):
        st.dataframe(df_long.head(200), width="stretch")

    st.divider()
    st.subheader("Demand Visualization")

    col_hourly, col_weekly = st.columns(2)

    with col_hourly:
        st.markdown("**Hourly Demand Trend**")
        hourly = df_long.groupby('hour')['volume_kwh'].mean()

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(hourly.index, hourly.values, color='#4FC3F7', linewidth=2, marker='o', markersize=3)
        ax.fill_between(hourly.index, hourly.values, alpha=0.1, color='#4FC3F7')
        ax.set_xlabel("Hour")
        ax.set_ylabel("Avg kWh")
        ax.set_xticks(range(0, 24, 1))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_weekly:
        st.markdown("**Weekday vs Weekend Demand**")
        df_long['day_type'] = df_long['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
        day_hour = df_long.groupby(['day_type', 'hour'])['volume_kwh'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(6, 3.5))
        for dtype, color in [('Weekday', '#4FC3F7'), ('Weekend', '#FF8A65')]:
            subset = day_hour[day_hour['day_type'] == dtype]
            ax.plot(subset['hour'], subset['volume_kwh'], color=color, linewidth=2, label=dtype)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Avg kWh")
        ax.set_xticks(range(0, 24, 1))
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    col_month, col_zones = st.columns(2)

    with col_month:
        st.markdown("**Monthly Demand**")
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
        st.markdown("**Top 10 High Demand Zones**")
        top_zones = (df_long.groupby('TAZID')['volume_kwh'].sum()
                     .nlargest(10).sort_values())

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(
            top_zones.index.astype(str),
            top_zones.values,
            color='#FF8A65',
        )
        ax.set_xlabel("Total kWh")
        ax.set_ylabel("Zone ID")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    col_model, col_btn = st.columns([2, 1])
    with col_model:
        model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])
    with col_btn:
        st.write("")
        run = st.button("Train and Predict", type="primary", use_container_width=True)

    if run:
        with st.spinner("Processing..."):
            model, X_train, X_test, y_train, y_test, feature_names = train_model(df_long, model_type)
            predictions, metrics = make_predictions(model, X_test, y_test)
            importances = get_feature_importance(model, feature_names)

        st.divider()
        st.subheader("Model Evaluation")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R2 Score", metrics['r2'])
        m2.metric("MAE", metrics['mae'])
        m3.metric("RMSE", metrics['rmse'])
        m4.metric("MAPE", f"{metrics['mape']}%")

        st.divider()
        st.subheader("Prediction Results")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_test.values[:200], color='#4FC3F7', linewidth=1.5, label='Actual')
        ax.plot(predictions[:200], color='#FF8A65', linewidth=1.5, linestyle='--', label='Predicted')
        ax.set_xlabel("Index")
        ax.set_ylabel("Volume (kWh)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_ai:
    st.header("Infrastructure Recommendations")
    st.markdown("This section uses an agent to analyze the demand data and local guidelines to suggest improvements.")
    
    if st.button("Generate Planning Report"):
        with st.spinner("Processing guidelines and data..."):
            top_zones_dict = (df_long.groupby('TAZID')['volume_kwh'].sum().nlargest(10).to_dict())
            peak_hour = df_long.groupby('hour')['volume_kwh'].mean().idxmax()
            
            demand_state = {
                "total_zones": summary['num_zones'],
                "average_kwh": summary['avg_volume'],
                "peak_hour": int(peak_hour),
                "high_load_zones": top_zones_dict
            }
            
            report = run_agentic_workflow(demand_state, api_key)
            st.session_state['report_content'] = report
            
            # Generate a summary chart for the AI report
            fig_rep, ax_rep = plt.subplots(figsize=(8, 4))
            top_zones_s = pd.Series(top_zones_dict)
            ax_rep.bar(top_zones_s.index.astype(str), top_zones_s.values, color='#81C784')
            ax_rep.set_title("Peak Demand Across Priority Expansion Zones")
            ax_rep.set_xlabel("Zone ID")
            ax_rep.set_ylabel("Total Energy (kWh)")
            plt.tight_layout()
            fig_rep.savefig(os.path.join(BASE_DIR, "data", "report_chart.png"))
            plt.close(fig_rep)

    if 'report_content' in st.session_state:
        st.image(os.path.join(BASE_DIR, "data", "report_chart.png"), caption="AI Generated Planning Context Chart")
        st.markdown(st.session_state['report_content'])
        
        pdf_bytes = generate_pdf_report(st.session_state['report_content'])
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
            file_name="EV_Plan_Report.pdf",
            mime="application/pdf",
        )

with tab_scenario:
    st.header("Growth Analysis")
    st.markdown("Compare the baseline demand with future growth scenarios based on EV adoption rates.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Future Load Projection")
        growth_rate = st.slider("Estimated Growth (%)", min_value=0, max_value=200, value=50, step=10)
        
        scenario_df = df_long.copy()
        scenario_df['projected_volume'] = scenario_df['volume_kwh'] * (1 + (growth_rate / 100))
        
        scenario_hourly = scenario_df.groupby('hour')[['volume_kwh', 'projected_volume']].mean()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(scenario_hourly.index, scenario_hourly['volume_kwh'], color='#4FC3F7', label='Baseline')
        ax.plot(scenario_hourly.index, scenario_hourly['projected_volume'], color='#E57373', linestyle='--', label='Projected')
        ax.set_xlabel("Hour")
        ax.set_ylabel("Avg kWh")
        ax.set_xticks(range(0, 24, 1))
        ax.legend()
        ax.set_title(f"Impact of {growth_rate}% Growth")
        st.pyplot(fig)
        plt.close(fig)
        
    with col_s2:
        st.subheader("Zone Comparison")
        all_zones = sorted(df_long['TAZID'].unique().tolist())
        selected_zones = st.multiselect("Select zones to compare:", all_zones, default=all_zones[:3])
        
        if len(selected_zones) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#4FC3F7', '#FFB74D', '#81C784', '#BA68C8']
            for i, zone in enumerate(selected_zones[:4]):
                zone_data = df_long[df_long['TAZID'] == zone].groupby('hour')['volume_kwh'].mean()
                ax.plot(zone_data.index, zone_data.values, color=colors[i%len(colors)], label=f"Zone {zone}")
                
            ax.set_xlabel("Hour")
            ax.set_ylabel("Avg kWh")
            ax.set_xticks(range(0, 24, 1))
            ax.legend()
            ax.set_title("Hourly Demand Comparison")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Select zones to see the comparison.")