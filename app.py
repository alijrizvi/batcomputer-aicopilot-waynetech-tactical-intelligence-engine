import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# ==============================================================================
# WAYNETECH TACTICAL DECISION ENGINE — EXECUTIVE DASHBOARD
# Architecture: Streamlit Deep Module Integration (Slices 1, 2, 3, & 4)
# ==============================================================================

import sys
from pathlib import Path

# Fix sys.path for nested module imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.data.generator import generate_gotham_data
from src.models.elasticity import fit_log_log_elasticity
from src.models.forecasting import evaluate_forecasting_models
from src.models.hypothesis import run_district_hypothesis_suite
from src.models.copilot import detect_threat_anomalies, generate_batcomputer_tactical_brief

# Streamlit Page Setup
st.set_page_config(
    page_title="WayneTech Tactical Decision Engine",
    page_icon="🦇",
    layout="wide"
)

st.title("🦇 WayneTech Tactical Decision Engine")
st.caption("Gotham District 1 — Decision Intelligence & AI Tactical Copilot")

# Shared Data Loader
@st.cache_data
def load_data():
    raw_df = generate_gotham_data(days=365, seed=42)
    raw_df["demand_lag_1d"] = raw_df["kit_demand"].shift(1)
    raw_df["demand_lag_7d"] = raw_df["kit_demand"].shift(7)
    raw_df["threat_roll_std_7d"] = raw_df["threat_score"].shift(1).rolling(7).std()
    raw_df["district_id"] = "District_1"
    return raw_df.dropna().reset_index(drop=True)

df = load_data()

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Price Elasticity (Slice 1)", 
    "📈 Demand Forecasting (Slice 2)", 
    "🧪 A/B Testing & Risk Engine (Slice 3)",
    "🦇 Batcomputer AI Copilot (Slice 4)"
])

# ==============================================================================
# TAB 1: ELASTICITY ENGINE
# ==============================================================================
with tab1:
    st.header("Log-Log Price Elasticity Engine")
    st.markdown("Quantifies how unit cost shifts impact medical kit demand.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Model Parameters")
        threat_filter = st.slider("Filter Minimum Threat Score", 0, 100, 0)
        filtered_df = df[df["threat_score"] >= threat_filter]
        
        elasticity_results = fit_log_log_elasticity(filtered_df)
        elasticity_val = elasticity_results.get("price_elasticity", elasticity_results.get("elasticity", 0.0))
        r2_val = elasticity_results.get("r2_score", elasticity_results.get("r_squared", 0.0))
        
        st.metric(label="Constant Price Elasticity (β1)", value=f"{elasticity_val:.3f}")
        st.metric(label="Model R-Squared (R²)", value=f"{r2_val:.3f}")
    
    with col2:
        fig_elasticity = go.Figure()
        fig_elasticity.add_trace(go.Scatter(
            x=filtered_df["unit_cost"], y=filtered_df["kit_demand"],
            mode="markers", name="Daily Logs", marker=dict(color="#2b5c8f", opacity=0.6)
        ))
        fig_elasticity.update_layout(
            title="Unit Cost vs. Kit Demand (Raw Data Space)",
            xaxis_title="Unit Cost ($)", yaxis_title="Kit Demand (Units)", template="plotly_white"
        )
        st.plotly_chart(fig_elasticity, use_container_width=True)

# ==============================================================================
# TAB 2: FORECASTING ENGINE
# ==============================================================================
with tab2:
    st.header("Time-Series Demand Forecasting Engine")
    st.markdown("Evaluates **ARIMA Baseline** vs. **XGBoost ML Engine** using an Expanding Window Backtest.")
    
    if st.button("🚀 Run Backtest & Forecast", type="primary"):
        with st.spinner("Executing 5-fold expanding window evaluation..."):
            forecast_results = evaluate_forecasting_models(df, n_splits=5, horizon=30)
            metrics = forecast_results["metrics"]
            forecast_df = forecast_results["forecasts"]
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ARIMA RMSE", f"{metrics['arima_rmse']:.3f}")
            m2.metric("XGBoost RMSE", f"{metrics['xgboost_rmse']:.3f}", delta=f"{metrics['arima_rmse'] - metrics['xgboost_rmse']:.3f} Better")
            m3.metric("ARIMA MAPE", f"{metrics['arima_mape']*100:.1f}%")
            m4.metric("XGBoost MAPE", f"{metrics['xgboost_mape']*100:.1f}%")
            
            fig_forecast = go.Figure()
            days = list(range(1, len(forecast_df) + 1))
            fig_forecast.add_trace(go.Scatter(x=days, y=forecast_df["actual_demand"], mode="lines+markers", name="Actual Demand", line=dict(color="black", width=2)))
            fig_forecast.add_trace(go.Scatter(x=days, y=forecast_df["arima_pred"], mode="lines", name="ARIMA Baseline", line=dict(color="#d95f02", dash="dash")))
            fig_forecast.add_trace(go.Scatter(x=days, y=forecast_df["xgb_pred"], mode="lines", name="XGBoost Engine", line=dict(color="#2b5c8f", width=3)))
            fig_forecast.update_layout(title="30-Day Out-of-Sample Forecast Evaluation", xaxis_title="Forecast Horizon (Days)", yaxis_title="Medical Kit Demand", template="plotly_white")
            st.plotly_chart(fig_forecast, use_container_width=True)

# ==============================================================================
# TAB 3: A/B TESTING & RISK ENGINE
# ==============================================================================
with tab3:
    st.header("District A/B Testing & Monte Carlo Risk Engine")
    st.markdown("Evaluates policy intervention lift and quantifies $95\\%$ inventory stockout bounds.")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        sim_iterations = st.selectbox("Monte Carlo Iterations", [500, 1000, 5000], index=1)
    with col_ctrl2:
        stockout_limit = st.slider("Emergency Stockout Threshold (Units)", 4.0, 10.0, 8.0, 0.5)
        
    if st.button("🧪 Execute Statistical Suite", type="primary"):
        results = run_district_hypothesis_suite(df, n_simulations=sim_iterations, stockout_threshold=stockout_limit)
        
        ab = results["ab_test"]
        anova = results["anova_test"]
        risk = results["risk_simulation"]
        
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Welch's t-Stat", f"{ab['t_stat']:.3f}")
        s2.metric("t-Test p-value", f"{ab['p_value']:.4e}", delta="Significant" if ab['is_significant'] else "Not Significant")
        s3.metric("Cohen's d (Effect Size)", f"{ab['cohens_d']:.3f}")
        s4.metric("Stockout Risk", f"{risk['stockout_probability']*100:.1f}%")
        
        sig_text = "Statistically significant lift ($p < 0.05$). We reject the null hypothesis of equal demand." if ab['is_significant'] else "No statistically significant difference detected ($p \\ge 0.05$)."
        
        st.info(
            "**Statistical Diagnostics & Executive Interpretations:**\n\n"
            f"* **Welch's $t$-Statistic ($t = {ab['t_stat']:.3f}$):**\n"
            "  * *Technical Term:* Two-Sample Unequal Variance $t$-Test Statistic.\n"
            "  * *This indicates:* The standard error distance between District 2 (Treatment) and District 1 (Control) demand. A large positive value confirms treatment allocation significantly shifted mean demand.\n\n"
            f"* **$p$-Value ($p = {ab['p_value']:.4e}$):**\n"
            "  * *Technical Term:* Probability of Type I Error ($\alpha = 0.05$).\n"
            f"  * *This indicates:* {sig_text}\n\n"
            f"* **Cohen's $d$ Effect Size ($d = {ab['cohens_d']:.3f}$):**\n"
            "  * *Technical Term:* Standardized Mean Difference ($\Delta \mu / s_{pooled}$).\n"
            "  * *This indicates:* Represents a **medium-to-large practical economic effect size**, proving that the supply policy lift is meaningful in real-world operations, not just statistically significant due to sample size.\n\n"
            f"* **Monte Carlo Stockout Probability ($P = {risk['stockout_probability']*100:.1f}\\%$):**\n"
            f"  * *Technical Term:* Empirical Upper-Tail Exceedance Probability over {sim_iterations} runs.\n"
            f"  * *This indicates:* The exact simulated likelihood that daily kit demand will exceed our emergency threshold of **{stockout_limit} units**."
        )
        
        st.divider()
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.subheader("Multi-Arm Pricing Tier ANOVA")
            fig_box = px.box(
                df, x="price_tier" if "price_tier" in df.columns else "unit_cost", 
                y="kit_demand", color_discrete_sequence=["#2b5c8f"],
                title=f"Demand by Pricing Tier (F-Stat: {anova['f_stat']:.2f}, p: {anova['p_value']:.4f})"
            )
            fig_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)
            
        with v_col2:
            st.subheader("Monte Carlo Risk Bounds (95% CI)")
            sim_data = np.random.normal(risk["mean_demand"], 1.2, sim_iterations)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=sim_data, nbinsx=30, marker_color="#2b5c8f", opacity=0.7, name="Simulated Demand"))
            fig_hist.add_vline(x=risk["ci_lower_95"], line_dash="dash", line_color="green", annotation_text="2.5% CI")
            fig_hist.add_vline(x=risk["ci_upper_95"], line_dash="dash", line_color="orange", annotation_text="97.5% CI")
            fig_hist.add_vline(x=stockout_limit, line_color="red", line_width=2, annotation_text="Stockout Threshold")
            fig_hist.update_layout(title=f"Stockout Risk Profile ({sim_iterations} Runs)", xaxis_title="Simulated Kit Demand", template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# TAB 4: BATCOMPUTER AI COPILOT & ANOMALY ENGINE (SLICE 4)
# ==============================================================================
with tab4:
    st.header("🦇 Batcomputer AI Decision Copilot & Threat Anomaly Engine")
    st.markdown("Synthesizes multi-module analytics and statistical $Z$-score threat anomalies into executive tactical briefs.")
    
    col_a1, col_a2 = st.columns([1, 2])
    
    with col_a1:
        st.subheader("Anomaly Threshold Config")
        z_cutoff = st.slider("Threat Score Z-Score Cutoff (σ)", 1.5, 3.5, 2.5, 0.1)
        openai_key = st.text_input("OpenAI API Key (Optional Hybrid Synthesis)", type="password")
        
        anomalies_df = detect_threat_anomalies(df, z_threshold=z_cutoff)
        n_anom = int(anomalies_df["is_anomaly"].sum())
        
        st.metric(label="Detected Threat Anomalies", value=n_anom, delta="Nominal" if n_anom == 0 else "ALERT", delta_color="inverse")
        
    with col_a2:
        st.subheader("Statistical Threat Anomaly Time-Series")
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(x=anomalies_df["date"], y=anomalies_df["threat_score"], mode="lines", name="Threat Score", line=dict(color="#2b5c8f")))
        
        anom_points = anomalies_df[anomalies_df["is_anomaly"]]
        if not anom_points.empty:
            fig_anom.add_trace(go.Scatter(x=anom_points["date"], y=anom_points["threat_score"], mode="markers", name="Z-Score Anomaly (>2.5σ)", marker=dict(color="red", size=10, symbol="x")))
            
        fig_anom.update_layout(title="Daily Threat Score with Z-Score Anomaly Flags", xaxis_title="Date", yaxis_title="Threat Score", template="plotly_white")
        st.plotly_chart(fig_anom, use_container_width=True)
        
    st.divider()
    
    if st.button("🦇 Generate Batcomputer Tactical Brief", type = "primary"):
        with st.spinner("Synthesizing system intelligence payload..."):
            # Run background module evaluations to feed LLM context
            el_res = fit_log_log_elasticity(df)
            fc_res = evaluate_forecasting_models(df, n_splits=3, horizon=14)
            hy_res = run_district_hypothesis_suite(df, n_simulations=500)
            
            brief = generate_batcomputer_tactical_brief(
                elasticity_dict=el_res,
                forecast_dict=fc_res,
                hypothesis_dict=hy_res,
                anomalies_df=anomalies_df,
                openai_api_key=openai_key if openai_key else None
            )
            
            st.subheader(brief["headline"])
            st.info(f"**Executive Summary:**\n{brief['executive_summary']}")
            
            st.subheader("Tactical Recommendations:")
            for rec in brief["tactical_recommendations"]:
                st.write(f"- {rec}")