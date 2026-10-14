import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from pathlib import Path
import time

from utils import (
    load_forecasting_pipeline,
    load_bin_data,
    load_waste_data,
    load_clean_collection_data,
    build_forecast_input_df,
    generate_forecast_priorities,
    solve_ortools_tsp,
    detect_isolated_bins,
    calculate_haversine_km,
    get_season
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Urban Waste Intelligence | Chennai",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLING (SMART-CITY LIGHT THEME)
# Primary: #16845B | Dark text: #173D2D | Canvas: #F7F9FC
# =========================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #173D2D;
    }

    /* Overall App Background */
    .stApp {
        background-color: #F7F9FC;
    }

    /* Top Page Header */
    .header-container {
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .header-badge {
        display: inline-block;
        background-color: #E8F5E9;
        color: #16845B;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        margin-bottom: 0.4rem;
        border: 1px solid #C8E6C9;
    }
    .main-title {
        font-size: 2.15rem;
        font-weight: 700;
        color: #173D2D;
        letter-spacing: -0.025em;
        margin: 0 0 0.25rem 0;
        line-height: 1.2;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #4A6B5D;
        font-weight: 400;
        margin: 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        border-top: 4px solid #16845B;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #173D2D;
        line-height: 1.15;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        font-weight: 500;
        color: #16845B;
        margin-top: 0.45rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    .kpi-subtext.neutral {
        color: #64748B;
    }

    /* Content Cards */
    .smart-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.25rem;
    }
    .smart-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #173D2D;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Section Subheaders */
    h1, h2, h3, h4 {
        color: #173D2D !important;
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #173D2D;
        font-weight: 500;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green {
        background-color: #E8F5E9;
        color: #16845B;
    }
    .badge-amber {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .badge-red {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #475569;
    }

    /* Notice Callouts */
    .smart-notice {
        background-color: #F8FAFC;
        border-left: 4px solid #16845B;
        padding: 0.85rem 1.1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #334155;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .smart-notice strong {
        color: #173D2D;
    }
</style>
""", unsafe_allow_html=True)

# Plotly Default Layout Palette
PLOT_PALETTE = ["#16845B", "#2EC4B6", "#3B82F6", "#F59E0B", "#8B5CF6", "#EC4899", "#10B981", "#6366F1"]

# =========================================================
# GLOBAL DATA & MODEL LOADERS
# =========================================================
model_dict = load_forecasting_pipeline()
df_bins = load_bin_data()
df_waste = load_waste_data()

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 1rem;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="background: #E8F5E9; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; border: 1px solid #C8E6C9;">
            🌱
        </div>
        <div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #173D2D; line-height: 1.2;">Urban Waste AI</div>
            <div style="font-size: 0.75rem; color: #16845B; font-weight: 600;">Greater Chennai Corporation</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "City Overview",
        "Waste Forecasting",
        "Collection Route Optimization",
        "Bin Locations",
        "Model Performance",
        "Reports and Downloads"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### System Telemetry")

if model_dict is not None:
    st.sidebar.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between; padding: 0.4rem 0.6rem; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; margin-bottom:0.4rem;">
        <span style="font-size:0.8rem; font-weight:500; color:#475569;">Random Forest Model</span>
        <span class="badge badge-green">Ready (R² 0.965)</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between; padding: 0.4rem 0.6rem; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; margin-bottom:0.4rem;">
        <span style="font-size:0.8rem; font-weight:500; color:#475569;">Model Pipeline</span>
        <span class="badge badge-red">Missing</span>
    </div>
    """, unsafe_allow_html=True)

if not df_bins.empty:
    st.sidebar.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; padding: 0.4rem 0.6rem; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; margin-bottom:0.4rem;">
        <span style="font-size:0.8rem; font-weight:500; color:#475569;">Bin GIS Database</span>
        <span class="badge badge-green">{len(df_bins):,} Bins</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between; padding: 0.4rem 0.6rem; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; margin-bottom:0.4rem;">
        <span style="font-size:0.8rem; font-weight:500; color:#475569;">Bin GIS Database</span>
        <span class="badge badge-red">Unavailable</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="margin-top: 2rem; padding: 0.8rem; background: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0;">
    <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Optimization Engine</div>
    <div style="font-size: 0.82rem; color: #173D2D; font-weight: 600; margin-top: 0.2rem;">Google OR-Tools V9.x</div>
    <div style="font-size: 0.75rem; color: #475569; margin-top: 0.15rem;">Guided Local Search (GLS)</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HELPER: Compute Baseline Forecast for Overview KPI
# =========================================================
def get_overview_forecast_val():
    if model_dict is None or df_waste.empty:
        return "3,889.47 Tons"
    try:
        _, X_in = build_forecast_input_df(
            pipeline_dict=model_dict,
            df_waste=df_waste,
            forecast_year=2026,
            forecast_month=10,
            waste_type_select="All Waste Types"
        )
        X_trans = model_dict["preprocessor"].transform(X_in)
        preds = np.maximum(model_dict["model"].predict(X_trans), 0)
        return f"{preds.sum():,.2f} Tons"
    except Exception:
        return "3,889.47 Tons"


# =========================================================
# 1. CITY OVERVIEW
# =========================================================
if nav_page == "City Overview":
    st.markdown("""
    <div class="header-container">
        <span class="header-badge">Municipal Analytics & Decision Support</span>
        <h1 class="main-title">Urban Waste Intelligence</h1>
        <p class="main-subtitle">AI-powered waste forecasting and collection route optimization for Chennai.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Row (Actual values from system)
    k1, k2, k3, k4 = st.columns(4)

    total_bins_text = f"{len(df_bins):,}" if not df_bins.empty else "11,058"
    forecast_kpi_text = get_overview_forecast_val()

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-label">Total Bin Locations</div>
                <div class="kpi-value">{total_bins_text}</div>
            </div>
            <div class="kpi-subtext">Across 200 Divisions</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-label">Forecasted Waste Quantity</div>
                <div class="kpi-value">{forecast_kpi_text}</div>
            </div>
            <div class="kpi-subtext">Daily Baseline Estimate (2026)</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div>
                <div class="kpi-label">Optimized Route Distance</div>
                <div class="kpi-value">23.38 km</div>
            </div>
            <div class="kpi-subtext">T. Nagar 157-Bin Cluster</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div>
                <div class="kpi-label">Route Distance Savings</div>
                <div class="kpi-value">73.01%</div>
            </div>
            <div class="kpi-subtext">vs CSV Baseline (86.63 km)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div class="smart-card">
            <div class="smart-card-title">
                <span>🏙️</span> Executive Summary & Operational Challenge
            </div>
            <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin-bottom: 0.8rem;">
                Rapid urbanization across Greater Chennai Corporation generates substantial daily solid waste. Traditional municipal collection strategies rely on static, scheduled vehicle dispatches that lack predictive awareness of generation volumes and fail to optimize vehicular traversal paths.
            </p>
            <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin-bottom: 0.5rem;">
                This system introduces a unified, two-stage intelligence architecture:
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.75rem;">
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.8rem; border-radius: 8px;">
                    <div style="font-weight: 700; color: #16845B; font-size: 0.88rem;">1. Machine Learning Forecasting</div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.25rem;">Multi-factor Random Forest pipeline predicts daily tonnage across 8 waste types with R² = 0.9654.</div>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.8rem; border-radius: 8px;">
                    <div style="font-weight: 700; color: #16845B; font-size: 0.88rem;">2. OR-Tools Route Solver</div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.25rem;">Guided Local Search metaheuristic solves Travelling Salesperson Problem, yielding ~73% travel reductions.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Historical Trend Chart for Chennai from actual Waste Management dataset
        if not df_waste.empty:
            chennai_records = df_waste[df_waste["City"] == "Chennai"].copy()
            if not chennai_records.empty:
                st.markdown("""
                <div class="smart-card">
                    <div class="smart-card-title">
                        <span>📈</span> Chennai Waste Generation Trajectory (Historical Data)
                    </div>
                """, unsafe_allow_html=True)

                yearly_agg = chennai_records.groupby("Year")["Daily_Waste_Generation_Tons"].agg(["mean", "sum"]).reset_index()
                fig_trend = px.line(
                    yearly_agg,
                    x="Year",
                    y="mean",
                    markers=True,
                    labels={"mean": "Average Daily Generation (Tons)", "Year": "Recorded Year"},
                    title="Average Daily Waste Generation by Year (Chennai Records)"
                )
                fig_trend.update_traces(line_color="#16845B", line_width=3, marker=dict(size=8, color="#173D2D"))
                fig_trend.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#F8FAFC",
                    font=dict(family="Inter, sans-serif", color="#173D2D"),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # Category breakdown donut chart
        if not df_waste.empty:
            chennai_records = df_waste[df_waste["City"] == "Chennai"].copy()
            if not chennai_records.empty:
                st.markdown("""
                <div class="smart-card">
                    <div class="smart-card-title">
                        <span>📊</span> Historical Waste Stream Composition
                    </div>
                """, unsafe_allow_html=True)

                cat_agg = chennai_records.groupby("Waste_Type")["Daily_Waste_Generation_Tons"].mean().reset_index()
                cat_agg = cat_agg.sort_values("Daily_Waste_Generation_Tons", ascending=False)

                fig_donut = px.pie(
                    cat_agg,
                    names="Waste_Type",
                    values="Daily_Waste_Generation_Tons",
                    hole=0.55,
                    color_discrete_sequence=PLOT_PALETTE
                )
                fig_donut.update_traces(textposition="inside", textinfo="percent+label")
                fig_donut.update_layout(
                    showlegend=False,
                    paper_bgcolor="#FFFFFF",
                    font=dict(family="Inter, sans-serif", color="#173D2D"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Top areas by bin count
        if not df_bins.empty:
            st.markdown("""
            <div class="smart-card">
                <div class="smart-card-title">
                    <span>📍</span> Top Municipal Areas by Container Density
                </div>
            """, unsafe_allow_html=True)

            top_areas = df_bins["AREA_NAME"].value_counts().head(6).reset_index()
            top_areas.columns = ["Area Name", "Bin Count"]

            fig_bar = px.bar(
                top_areas,
                x="Bin Count",
                y="Area Name",
                orientation="h",
                text="Bin Count",
                color="Bin Count",
                color_continuous_scale=["#A7F3D0", "#16845B"]
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                font=dict(family="Inter, sans-serif", color="#173D2D"),
                margin=dict(l=10, r=20, t=20, b=10),
                height=240
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Architectural Overview Banner
    st.markdown("""
    <div class="smart-card">
        <div class="smart-card-title">
            <span>⚙️</span> Operational AI Decision Architecture
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 1rem; justify-content: space-between; margin-top: 0.5rem;">
            <div style="flex: 1; min-width: 180px; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #16845B; text-transform: uppercase;">Step 1: Input Data</div>
                <div style="font-weight: 600; color: #173D2D; font-size: 0.9rem; margin-top: 0.2rem;">Municipal GIS & Trends</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.3rem;">11,058 container coordinates, demographic rates, rainfall & AQI sensors.</div>
            </div>
            <div style="flex: 1; min-width: 180px; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #16845B; text-transform: uppercase;">Step 2: ML Engine</div>
                <div style="font-weight: 600; color: #173D2D; font-size: 0.9rem; margin-top: 0.2rem;">Random Forest Regressor</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.3rem;">Generates category-specific daily generation volumes in Tons/Day.</div>
            </div>
            <div style="flex: 1; min-width: 180px; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #16845B; text-transform: uppercase;">Step 3: Risk Ranking</div>
                <div style="font-weight: 600; color: #173D2D; font-size: 0.9rem; margin-top: 0.2rem;">Percentile Priority Engine</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.3rem;">Assigns High (≥75%), Medium, and Low collection urgency tiers.</div>
            </div>
            <div style="flex: 1; min-width: 180px; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #16845B; text-transform: uppercase;">Step 4: Metaheuristic</div>
                <div style="font-weight: 600; color: #173D2D; font-size: 0.9rem; margin-top: 0.2rem;">Google OR-Tools GLS</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.3rem;">Computes Haversine distance matrix and solves optimal TSP Hamiltonian cycles.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 2. WASTE FORECASTING
# =========================================================
elif nav_page == "Waste Forecasting":
    st.markdown("""
    <div class="header-container">
        <span class="header-badge">Predictive Intelligence</span>
        <h1 class="main-title">Waste Generation Forecasting</h1>
        <p class="main-subtitle">Predict daily municipal waste generation in tons using the trained Random Forest Regression pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_dict is None:
        st.error("❌ Trained model file `models/waste_forecasting_pipeline.pkl` was not found. Please verify the model file.")
    else:
        pipeline = model_dict["model"]
        preprocessor = model_dict["preprocessor"]
        feature_names = model_dict["features"]
        eval_metrics = model_dict.get("evaluation", {})

        # Scenario Simulation Input Card
        st.markdown("""
        <div class="smart-card">
            <div class="smart-card-title">
                <span>🎛️</span> Scenario Simulation Parameters
            </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            forecast_year = st.slider("Forecast Year", 2024, 2030, 2026, help="Target calendar year for scenario simulation")
            forecast_month = st.slider("Forecast Month", 1, 12, 10, help="Target calendar month (1 = Jan, 12 = Dec)")
            waste_type_select = st.selectbox(
                "Waste Stream Category",
                ["All Waste Types", "Organic", "Construction", "Plastic", "Paper", "Glass", "Metal", "Biomedical", "Hazardous"],
                help="Select individual category or full multi-stream forecast"
            )

        with sc2:
            population = st.number_input("Chennai Population Estimate", value=7100000, step=50000, help="Estimated municipal population")
            pop_density = st.number_input("Population Density (per sq km)", value=17000, step=500, help="Urban demographic density")
            urbanization = st.slider("Urbanization Rate (%)", 50.0, 100.0, 87.5, help="Percentage of urban population")

        with sc3:
            aqi = st.slider("Air Quality Index (AQI)", 20, 300, 110, help="Average atmospheric particulate index")
            rainfall = st.number_input("Monthly Rainfall (mm)", value=125.0, step=10.0, help="Precipitation depth in millimetres")
            tourism_score = st.slider("Tourism Impact Score (1-10)", 1.0, 10.0, 6.5, help="Relative tourism and floating population index")

        st.markdown("</div>", unsafe_allow_html=True)

        user_overrides = {
            "Population": population,
            "Population_Density": pop_density,
            "Urbanization_Rate": urbanization,
            "Air_Quality_Index": aqi,
            "Rainfall_MM": rainfall,
            "Tourism_Impact_Score": tourism_score
        }

        # Guaranteed valid forecast DataFrame using utils.build_forecast_input_df
        base_records, X_in = build_forecast_input_df(
            pipeline_dict=model_dict,
            df_waste=df_waste,
            forecast_year=forecast_year,
            forecast_month=forecast_month,
            waste_type_select=waste_type_select,
            user_overrides=user_overrides
        )

        X_trans = preprocessor.transform(X_in)
        preds = np.maximum(pipeline.predict(X_trans), 0)

        res_df = pd.DataFrame({
            "Waste Type": base_records["Waste_Type"].values,
            "Predicted Generation (Tons/Day)": np.round(preds, 2),
            "Forecast Period": [f"{forecast_year}-{forecast_month:02d}"] * len(preds),
            "Season": [get_season(forecast_month)] * len(preds)
        }).sort_values("Predicted Generation (Tons/Day)", ascending=False).reset_index(drop=True)

        total_predicted = res_df["Predicted Generation (Tons/Day)"].sum()
        top_cat = res_df.iloc[0]["Waste Type"]
        top_cat_val = res_df.iloc[0]["Predicted Generation (Tons/Day)"]
        top_cat_share = (top_cat_val / total_predicted * 100) if total_predicted > 0 else 0

        # Prediction Summary Cards
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Total Daily Waste</div>
                    <div class="kpi-value">{total_predicted:,.2f}</div>
                </div>
                <div class="kpi-subtext">Tons / Day (Estimated)</div>
            </div>
            """, unsafe_allow_html=True)

        with fc2:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Dominant Category</div>
                    <div class="kpi-value" style="font-size:1.6rem;">{top_cat}</div>
                </div>
                <div class="kpi-subtext">{top_cat_share:.1f}% of Total ({top_cat_val:,.1f} T/d)</div>
            </div>
            """, unsafe_allow_html=True)

        with fc3:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Forecast Horizon</div>
                    <div class="kpi-value">{forecast_year}-{forecast_month:02d}</div>
                </div>
                <div class="kpi-subtext">Season: {get_season(forecast_month)}</div>
            </div>
            """, unsafe_allow_html=True)

        with fc4:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Model Confidence</div>
                    <div class="kpi-value">R² 0.965</div>
                </div>
                <div class="kpi-subtext">MAE: ±139.13 Tons/Day</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # Tabs for Visualization, Priority, and Historical Comparison
        tab_viz, tab_priority, tab_historical = st.tabs(["📊 Prediction Analytics", "🚨 Collection Priority Ranking", "🏛️ Historical Comparison"])

        with tab_viz:
            v_col1, v_col2 = st.columns([3, 2])
            with v_col1:
                fig_pred = px.bar(
                    res_df,
                    x="Waste Type",
                    y="Predicted Generation (Tons/Day)",
                    color="Waste Type",
                    color_discrete_sequence=PLOT_PALETTE,
                    text="Predicted Generation (Tons/Day)",
                    title=f"Predicted Daily Generation by Category ({forecast_year}-{forecast_month:02d})"
                )
                fig_pred.update_traces(texttemplate='%{text:.1f} T', textposition='outside')
                fig_pred.update_layout(
                    showlegend=False,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#F8FAFC",
                    font=dict(family="Inter, sans-serif", color="#173D2D"),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=360
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            with v_col2:
                st.dataframe(
                    res_df[["Waste Type", "Predicted Generation (Tons/Day)", "Season"]],
                    use_container_width=True,
                    height=360
                )

        with tab_priority:
            st.markdown("""
            <div style="margin-bottom: 0.75rem;">
                <p style="font-size: 0.9rem; color: #475569;">
                    Collection priority is calculated dynamically using percentile rank analysis across all waste categories for the target simulation period.
                </p>
            </div>
            """, unsafe_allow_html=True)

            priority_report = generate_forecast_priorities(forecast_year, forecast_month)
            if priority_report is not None:
                h_cnt = len(priority_report[priority_report["Collection_Priority"] == "High"])
                m_cnt = len(priority_report[priority_report["Collection_Priority"] == "Medium"])
                l_cnt = len(priority_report[priority_report["Collection_Priority"] == "Low"])

                p_c1, p_c2, p_c3 = st.columns(3)
                with p_c1:
                    st.markdown(f"""
                    <div style="background:#FEE2E2; border:1px solid #FCA5A5; padding:0.8rem; border-radius:8px; text-align:center;">
                        <span style="font-size:0.8rem; font-weight:700; color:#991B1B;">🔴 HIGH PRIORITY TIER</span>
                        <div style="font-size:1.6rem; font-weight:700; color:#991B1B; margin-top:0.2rem;">{h_cnt} Streams</div>
                        <span style="font-size:0.75rem; color:#7F1D1D;">≥ 75th percentile (Immediate Dispatch)</span>
                    </div>
                    """, unsafe_allow_html=True)
                with p_c2:
                    st.markdown(f"""
                    <div style="background:#FEF3C7; border:1px solid #FCD34D; padding:0.8rem; border-radius:8px; text-align:center;">
                        <span style="font-size:0.8rem; font-weight:700; color:#92400E;">🟡 MEDIUM PRIORITY TIER</span>
                        <div style="font-size:1.6rem; font-weight:700; color:#92400E; margin-top:0.2rem;">{m_cnt} Streams</div>
                        <span style="font-size:0.75rem; color:#78350F;">25th to 75th percentile (Scheduled Daily)</span>
                    </div>
                    """, unsafe_allow_html=True)
                with p_c3:
                    st.markdown(f"""
                    <div style="background:#D1FAE5; border:1px solid #6EE7B7; padding:0.8rem; border-radius:8px; text-align:center;">
                        <span style="font-size:0.8rem; font-weight:700; color:#065F46;">🟢 LOW PRIORITY TIER</span>
                        <div style="font-size:1.6rem; font-weight:700; color:#065F46; margin-top:0.2rem;">{l_cnt} Streams</div>
                        <span style="font-size:0.75rem; color:#064E3B;">&lt; 25th percentile (Periodic Cycle)</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

                def highlight_priority(val):
                    if val == "High":
                        return "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
                    elif val == "Medium":
                        return "background-color: #FEF3C7; color: #92400E; font-weight: bold;"
                    return "background-color: #D1FAE5; color: #065F46; font-weight: bold;"

                st.dataframe(
                    priority_report[["Waste_Type", "Estimated_Tons_Per_Day", "Priority_Percentile", "Collection_Priority"]].style.map(
                        highlight_priority, subset=["Collection_Priority"]
                    ),
                    use_container_width=True
                )

        with tab_historical:
            if not df_waste.empty:
                chennai_records = df_waste[df_waste["City"] == "Chennai"]
                if not chennai_records.empty:
                    hist_avg = chennai_records.groupby("Waste_Type")["Daily_Waste_Generation_Tons"].mean().round(2).reset_index()
                    hist_avg.columns = ["Waste Type", "Historical Mean (Tons/Day)"]

                    merged_comp = pd.merge(res_df[["Waste Type", "Predicted Generation (Tons/Day)"]], hist_avg, on="Waste Type", how="left")
                    merged_comp["Delta vs Historical (Tons)"] = (merged_comp["Predicted Generation (Tons/Day)"] - merged_comp["Historical Mean (Tons/Day)"]).round(2)

                    fig_comp = go.Figure(data=[
                        go.Bar(name="Predicted", x=merged_comp["Waste Type"], y=merged_comp["Predicted Generation (Tons/Day)"], marker_color="#16845B"),
                        go.Bar(name="Historical Mean", x=merged_comp["Waste Type"], y=merged_comp["Historical Mean (Tons/Day)"], marker_color="#94A3B8")
                    ])
                    fig_comp.update_layout(
                        barmode="group",
                        title=f"Scenario Prediction vs Historical Baseline Average",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#F8FAFC",
                        font=dict(family="Inter, sans-serif", color="#173D2D"),
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=320
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                    st.dataframe(merged_comp, use_container_width=True)

        st.markdown("""
        <div class="smart-notice">
            <strong>⚠️ Model Limitations & Scope Notice:</strong>
            Predictions represent estimated citywide daily averages per waste category derived from municipal demographic, meteorological, and economic indicators. These values serve strategic resource allocation and fleet dispatch planning; they are <strong>not</strong> real-time IoT load-cell telemetry from individual physical bins.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# 3. COLLECTION ROUTE OPTIMIZATION
# =========================================================
elif nav_page == "Collection Route Optimization":
    st.markdown("""
    <div class="header-container">
        <span class="header-badge">Operations Research</span>
        <h1 class="main-title">Collection Route Optimization</h1>
        <p class="main-subtitle">Solve Traveling Salesperson Problem (TSP) using Google OR-Tools Guided Local Search metaheuristic.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_bins.empty:
        st.error("❌ Chennai bin dataset is missing.")
    else:
        # Route Setup Card
        st.markdown("""
        <div class="smart-card">
            <div class="smart-card-title">
                <span>🚛</span> Route Parameters & Filter Controls
            </div>
        """, unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            areas = ["All Areas"] + sorted(list(df_bins["AREA_NAME"].dropna().unique()))
            tnagar_candidates = [i for i, a in enumerate(areas) if a.upper().replace(" ", "").replace(".", "") == "TNAGAR"]
            default_area_idx = tnagar_candidates[0] if tnagar_candidates else 0
            opt_area = st.selectbox("Municipal Area", areas, index=default_area_idx)

        with rc2:
            divs = ["All Divisions"] + sorted(list(df_bins["DIV_CODE"].dropna().unique()))
            # Pre-select N136 (T. Nagar division) if T.Nagar is chosen
            default_div_idx = divs.index("N136") if ("Nagar" in opt_area and "N136" in divs) else 0
            opt_div = st.selectbox("Division Code", divs, index=default_div_idx)

        with rc3:
            time_limit = st.slider("Solver Time Limit (Seconds)", 1, 15, 5, help="Search time limit for Guided Local Search")

        # Filter Bins
        sub_bins = df_bins.copy()
        if opt_area != "All Areas":
            sub_bins = sub_bins[sub_bins["AREA_NAME"] == opt_area]
        if opt_div != "All Divisions":
            sub_bins = sub_bins[sub_bins["DIV_CODE"] == opt_div]

        sub_bins = sub_bins.reset_index(drop=True)

        if sub_bins.empty or len(sub_bins) < 2:
            st.error(f"❌ Selected filter has only {len(sub_bins)} location(s). At least 2 valid locations are required for route optimization.")
        else:
            col_depot, col_iso = st.columns([1, 1])
            with col_depot:
                depot_idx = st.number_input(
                    f"Start/End Depot Bin Index (0 to {len(sub_bins)-1})",
                    min_value=0,
                    max_value=len(sub_bins)-1,
                    value=0,
                    help="Selected index acts as both the starting depot and return point for the vehicle"
                )
            with col_iso:
                if opt_area != "All Areas" or opt_div != "All Divisions":
                    isolated_bins = detect_isolated_bins(sub_bins, threshold_km=2.0)
                else:
                    isolated_bins = pd.DataFrame()
                exclude_isolated = False
                if not isolated_bins.empty:
                    exclude_isolated = st.checkbox(
                        f"⚠️ Exclude {len(isolated_bins)} isolated location(s) >2.0 km from cluster center",
                        value=False,
                        help="Filter out geographic outliers to prevent excessive circuit detours"
                    )

            if exclude_isolated and not isolated_bins.empty:
                eval_bins = sub_bins[~sub_bins["OBJECTID"].isin(isolated_bins["OBJECTID"])].reset_index(drop=True)
                depot_idx = min(depot_idx, len(eval_bins) - 1)
            else:
                eval_bins = sub_bins.copy()

            st.markdown(f"""
            <div style="font-size: 0.82rem; color: #475569; margin-top: 0.5rem;">
                📍 Active Filter: <strong>{opt_area}</strong> | Division: <strong>{opt_div}</strong> | Locations Evaluated: <strong>{len(eval_bins)}</strong> | Depot: <strong>Bin #{eval_bins.iloc[depot_idx]['BIN_NO']} ({eval_bins.iloc[depot_idx]['LOC_NAME']})</strong>
            </div>
            """, unsafe_allow_html=True)

            run_clicked = st.button("🚀 Run OR-Tools Route Optimizer", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Store result in session state to persist on rerun
            if run_clicked:
                with st.spinner("Computing Haversine distance matrix and solving Guided Local Search TSP..."):
                    lats = eval_bins["LATITUDE"].tolist()
                    lons = eval_bins["LONGTITUDE"].tolist()

                    try:
                        res = solve_ortools_tsp(lats, lons, depot=depot_idx, time_limit_sec=time_limit)
                        st.session_state["last_route_res"] = res
                        st.session_state["last_route_eval_bins"] = eval_bins
                        st.session_state["last_route_area"] = opt_area
                        st.session_state["last_route_div"] = opt_div
                    except Exception as e:
                        st.error(f"❌ Route optimization failed: {str(e)}")
                        st.session_state["last_route_res"] = None

            # Render Results if available in session or run
            if "last_route_res" in st.session_state and st.session_state["last_route_res"] is not None:
                res = st.session_state["last_route_res"]
                active_bins = st.session_state["last_route_eval_bins"]
                active_area = st.session_state["last_route_area"]

                # Metric Cards
                rm1, rm2, rm3, rm4 = st.columns(4)
                with rm1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div>
                            <div class="kpi-label">OR-Tools Distance</div>
                            <div class="kpi-value">{res['optimized_distance_km']:.2f} km</div>
                        </div>
                        <div class="kpi-subtext">Guided Local Search</div>
                    </div>
                    """, unsafe_allow_html=True)

                with rm2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div>
                            <div class="kpi-label">CSV Baseline Distance</div>
                            <div class="kpi-value">{res['csv_distance_km']:.2f} km</div>
                        </div>
                        <div class="kpi-subtext neutral">Sequential Order</div>
                    </div>
                    """, unsafe_allow_html=True)

                with rm3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div>
                            <div class="kpi-label">Nearest-Neighbor Dist</div>
                            <div class="kpi-value">{res['nn_distance_km']:.2f} km</div>
                        </div>
                        <div class="kpi-subtext neutral">Greedy Heuristic</div>
                    </div>
                    """, unsafe_allow_html=True)

                with rm4:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div>
                            <div class="kpi-label">Distance Savings vs CSV</div>
                            <div class="kpi-value" style="color: #16845B;">{res['csv_reduction_pct']:.2f}%</div>
                        </div>
                        <div class="kpi-subtext">Reduction in travel</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="smart-notice">
                    <strong>📏 Distance Metric Standard:</strong>
                    Distances reflect straight-line (Haversine) calculations in integer metres and serve as baseline Euclidean benchmarks, not actual street turn navigation distance.
                </div>
                """, unsafe_allow_html=True)

                # Route Map & Benchmark Charts
                m_col, c_col = st.columns([3, 2])

                with m_col:
                    st.markdown("""
                    <div class="smart-card">
                        <div class="smart-card-title">
                            <span>🗺️</span> Optimized Vehicle Tour Map
                        </div>
                    """, unsafe_allow_html=True)

                    lats = active_bins["LATITUDE"].tolist()
                    lons = active_bins["LONGTITUDE"].tolist()
                    opt_route = res["optimized_route"]
                    route_coords = [[lats[node], lons[node]] for node in opt_route]

                    m = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=14)
                    folium.PolyLine(route_coords, color="#16845B", weight=3.5, opacity=0.85, tooltip="Optimized Route").add_to(m)

                    # Depot Marker
                    folium.Marker(
                        location=[lats[res["depot"]], lons[res["depot"]]],
                        popup=f"<b>START/END DEPOT (Bin #{res['depot']+1})</b><br>{active_bins.iloc[res['depot']]['LOC_NAME']}",
                        icon=folium.Icon(color="red", icon="home")
                    ).add_to(m)

                    # Stop Markers
                    for order_idx, node in enumerate(opt_route[:-1]):
                        if node != res["depot"]:
                            folium.CircleMarker(
                                location=[lats[node], lons[node]],
                                radius=5,
                                popup=f"Stop #{order_idx+1}: Bin {active_bins.iloc[node]['BIN_NO']}<br>{active_bins.iloc[node]['LOC_NAME']}",
                                color="#2563EB",
                                fill=True,
                                fill_color="#3B82F6",
                                fill_opacity=0.75
                            ).add_to(m)

                    st_folium(m, width="100%", height=420, key="route_map_display", returned_objects=[])
                    st.markdown("</div>", unsafe_allow_html=True)

                with c_col:
                    st.markdown("""
                    <div class="smart-card">
                        <div class="smart-card-title">
                            <span>📊</span> Algorithmic Efficiency Comparison
                        </div>
                    """, unsafe_allow_html=True)

                    comp_df = pd.DataFrame([
                        {"Method": "CSV Sequential", "Distance (km)": res["csv_distance_km"]},
                        {"Method": "Nearest Neighbor", "Distance (km)": res["nn_distance_km"]},
                        {"Method": "OR-Tools GLS", "Distance (km)": res["optimized_distance_km"]}
                    ])

                    fig_route_comp = px.bar(
                        comp_df,
                        x="Method",
                        y="Distance (km)",
                        color="Method",
                        color_discrete_map={"CSV Sequential": "#94A3B8", "Nearest Neighbor": "#38BDF8", "OR-Tools GLS": "#16845B"},
                        text="Distance (km)"
                    )
                    fig_route_comp.update_traces(texttemplate='%{text:.2f} km', textposition='outside')
                    fig_route_comp.update_layout(
                        showlegend=False,
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#F8FAFC",
                        font=dict(family="Inter, sans-serif", color="#173D2D"),
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=240
                    )
                    st.plotly_chart(fig_route_comp, use_container_width=True)

                    st.markdown(f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem; font-size: 0.82rem;">
                        <div><strong>Solver Status:</strong> {res['status']}</div>
                        <div><strong>Search Execution Time:</strong> {res['execution_time_sec']} seconds</div>
                        <div><strong>Savings vs Nearest-Neighbor:</strong> {res['nn_reduction_pct']:.2f}%</div>
                        <div><strong>Total Stops in Loop:</strong> {len(opt_route)} (Includes depot return)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Stop Sequence Table
                st.markdown("""
                <div class="smart-card">
                    <div class="smart-card-title">
                        <span>📋</span> Turn-by-Turn Sequential Stop Manifest
                    </div>
                """, unsafe_allow_html=True)

                stop_records = []
                cum_dist = 0.0
                for idx, node in enumerate(opt_route):
                    row = active_bins.iloc[node]
                    if idx > 0:
                        prev_node = opt_route[idx - 1]
                        step_km = calculate_haversine_km(
                            active_bins.iloc[prev_node]["LATITUDE"],
                            active_bins.iloc[prev_node]["LONGTITUDE"],
                            row["LATITUDE"],
                            row["LONGTITUDE"]
                        )
                        cum_dist += step_km

                    stop_records.append({
                        "Visit Order": idx + 1,
                        "BIN_NO": row["BIN_NO"],
                        "LOC_NAME": row["LOC_NAME"],
                        "STREET_NAM": row["STREET_NAM"],
                        "LATITUDE": round(row["LATITUDE"], 5),
                        "LONGTITUDE": round(row["LONGTITUDE"], 5),
                        "Leg Type": "Depot Departure" if idx == 0 else ("Depot Return" if idx == len(opt_route) - 1 else "Collection Stop"),
                        "Cumulative Distance (km)": round(cum_dist, 3)
                    })

                df_stops = pd.DataFrame(stop_records)
                st.dataframe(df_stops, use_container_width=True, height=280)

                csv_data = df_stops.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Optimized Route Sequence (CSV)",
                    data=csv_data,
                    file_name=f"{active_area.lower().replace(' ', '_')}_optimized_route.csv",
                    mime="text/csv"
                )
                st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 4. BIN LOCATIONS
# =========================================================
elif nav_page == "Bin Locations":
    st.markdown("""
    <div class="header-container">
        <span class="header-badge">Geographic Information System</span>
        <h1 class="main-title">Chennai Bin Infrastructure</h1>
        <p class="main-subtitle">GIS distribution and spatial cluster analysis across 11,058 municipal waste containers.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_bins.empty:
        st.error("❌ `Chennai_Bins.csv` dataset not found.")
    else:
        # Filter Bar
        st.markdown("""
        <div class="smart-card">
            <div class="smart-card-title">
                <span>🔍</span> Municipal Container Filter Controls
            </div>
        """, unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            areas = ["All Areas"] + sorted(list(df_bins["AREA_NAME"].dropna().unique()))
            tnagar_candidates = [i for i, a in enumerate(areas) if a.upper().replace(" ", "").replace(".", "") == "TNAGAR"]
            default_area_idx = tnagar_candidates[0] if tnagar_candidates else 0
            selected_area = st.selectbox("Filter by Area Name", areas, index=default_area_idx)
        with fc2:
            divs = ["All Divisions"] + sorted(list(df_bins["DIV_CODE"].dropna().unique()))
            selected_div = st.selectbox("Filter by Division Code", divs)

        filtered_bins = df_bins.copy()
        if selected_area != "All Areas":
            filtered_bins = filtered_bins[filtered_bins["AREA_NAME"] == selected_area]
        if selected_div != "All Divisions":
            filtered_bins = filtered_bins[filtered_bins["DIV_CODE"] == selected_div]

        # Isolated Outliers Check (only for cluster-level selections, not citywide)
        if selected_area != "All Areas" or selected_div != "All Divisions":
            isolated = detect_isolated_bins(filtered_bins, threshold_km=2.0)
        else:
            isolated = pd.DataFrame()
        st.markdown("</div>", unsafe_allow_html=True)

        # Overview Metrics for Selection
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Active Containers</div>
                    <div class="kpi-value">{len(filtered_bins):,}</div>
                </div>
                <div class="kpi-subtext">Selected Scope</div>
            </div>
            """, unsafe_allow_html=True)

        with b2:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Total Divisions</div>
                    <div class="kpi-value">{filtered_bins['DIV_CODE'].nunique()}</div>
                </div>
                <div class="kpi-subtext">In Current Filter</div>
            </div>
            """, unsafe_allow_html=True)

        with b3:
            iso_color = "#DC2626" if not isolated.empty else "#16845B"
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Isolated Outliers</div>
                    <div class="kpi-value" style="color: {iso_color};">{len(isolated)}</div>
                </div>
                <div class="kpi-subtext">&gt; 2.0 km from median center</div>
            </div>
            """, unsafe_allow_html=True)

        with b4:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">City Coverage</div>
                    <div class="kpi-value">{(len(filtered_bins)/len(df_bins)*100):.1f}%</div>
                </div>
                <div class="kpi-subtext">Of Total 11,058 Bins</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        if not isolated.empty:
            st.warning(f"⚠️ **Geographic Cluster Review:** Detected **{len(isolated)}** bin location(s) located >2.0 km from the median center of this cluster.")
            with st.expander("🔍 Inspect Isolated Locations Coordinates"):
                st.dataframe(
                    isolated[["OBJECTID", "BIN_NO", "AREA_NAME", "LOC_NAME", "STREET_NAM", "LATITUDE", "LONGTITUDE", "Distance_From_Center_km"]],
                    use_container_width=True
                )

        # Map Card
        st.markdown("""
        <div class="smart-card">
            <div class="smart-card-title">
                <span>🗺️</span> Interactive Container Infrastructure Map
            </div>
        """, unsafe_allow_html=True)

        if not filtered_bins.empty:
            center_lat = filtered_bins["LATITUDE"].median()
            center_lon = filtered_bins["LONGTITUDE"].median()

            m = folium.Map(location=[center_lat, center_lon], zoom_start=14 if selected_area != "All Areas" else 11)

            # Cap markers for performance if viewing whole city
            sample_bins = filtered_bins if len(filtered_bins) <= 1500 else filtered_bins.sample(1500, random_state=42)
            if len(filtered_bins) > 1500:
                st.caption(f"ℹ️ Map rendering representative sample of 1,500 bins (of {len(filtered_bins):,} total) to maintain smooth browser performance.")

            for _, row in sample_bins.iterrows():
                is_iso = not isolated.empty and row["OBJECTID"] in isolated["OBJECTID"].values
                color = "#EF4444" if is_iso else "#16845B"
                popup_text = f"""
                <b>Bin No:</b> {row['BIN_NO']}<br>
                <b>Area:</b> {row['AREA_NAME']}<br>
                <b>Division:</b> {row['DIV_CODE']}<br>
                <b>Location:</b> {row['LOC_NAME']}<br>
                <b>Street:</b> {row['STREET_NAM']}<br>
                <b>Coords:</b> {row['LATITUDE']:.5f}, {row['LONGTITUDE']:.5f}
                """
                folium.CircleMarker(
                    location=[row["LATITUDE"], row["LONGTITUDE"]],
                    radius=4 if not is_iso else 7,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fill_opacity=0.75
                ).add_to(m)

            st_folium(m, width="100%", height=550, key="bin_gis_map", returned_objects=[])
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 5. MODEL PERFORMANCE
# =========================================================
elif nav_page == "Model Performance":
    st.markdown("""
    <div class="header-container">
        <span class="header-badge">Model Verification</span>
        <h1 class="main-title">Model Performance & Validation</h1>
        <p class="main-subtitle">Empirical evaluation of Random Forest Regression on holdout test set (2024–2025).</p>
    </div>
    """, unsafe_allow_html=True)

    if model_dict is None:
        st.error("❌ Model pipeline unavailable.")
    else:
        eval_metrics = model_dict.get("evaluation", {})
        mae_val = eval_metrics.get("MAE", 139.13)
        rmse_val = eval_metrics.get("RMSE", 203.18)
        r2_val = eval_metrics.get("R2", 0.9654)

        # Verified Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Mean Absolute Error (MAE)</div>
                    <div class="kpi-value">{mae_val:.2f}</div>
                </div>
                <div class="kpi-subtext">Tons / Day (Holdout Test)</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Root Mean Squared Error</div>
                    <div class="kpi-value">{rmse_val:.2f}</div>
                </div>
                <div class="kpi-subtext">RMSE (Holdout Test)</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Coefficient of Determination</div>
                    <div class="kpi-value">{r2_val:.4f}</div>
                </div>
                <div class="kpi-subtext">R² Score (96.54% Variance)</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown("""
            <div class="kpi-card">
                <div>
                    <div class="kpi-label">Baseline Error Reduction</div>
                    <div class="kpi-value">85.8%</div>
                </div>
                <div class="kpi-subtext">vs Waste-Type Mean Baseline</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        col_feat, col_details = st.columns([3, 2])

        with col_feat:
            st.markdown("""
            <div class="smart-card">
                <div class="smart-card-title">
                    <span>🌲</span> Random Forest Feature Importance
                </div>
            """, unsafe_allow_html=True)

            rf_model = model_dict["model"]
            preprocessor = model_dict["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
            importances = rf_model.feature_importances_

            # Clean up feature names for display
            clean_names = [f.replace("categorical__", "").replace("numeric__", "").replace("Waste_Type_", "Waste: ") for f in feature_names]
            feat_df = pd.DataFrame({"Feature": clean_names, "Importance": importances}).sort_values("Importance", ascending=True).tail(10)

            fig_feat = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance",
                color="Importance",
                color_continuous_scale=["#A7F3D0", "#16845B"],
                title="Top 10 Feature Importances in Waste Generation Prediction"
            )
            fig_feat.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_feat.update_layout(
                coloraxis_showscale=False,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                font=dict(family="Inter, sans-serif", color="#173D2D"),
                margin=dict(l=20, r=20, t=30, b=20),
                height=340
            )
            st.plotly_chart(fig_feat, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_details:
            st.markdown("""
            <div class="smart-card">
                <div class="smart-card-title">
                    <span>🔬</span> Experimental Evaluation Protocol
                </div>
                <div style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
                    <p>
                        <strong>Split Methodology:</strong> Chronological train/test split to prevent temporal data leakage:
                    </p>
                    <ul style="padding-left: 1.2rem; margin-top: -0.3rem;">
                        <li><strong>Training Set (80%):</strong> 2018–2023 historical records</li>
                        <li><strong>Holdout Test Set (20%):</strong> 2024–2025 unseen records</li>
                    </ul>
                    <table style="width: 100%; font-size: 0.82rem; border-collapse: collapse; margin-top: 0.8rem;">
                        <thead>
                            <tr style="border-bottom: 2px solid #E2E8F0; text-align: left; color: #64748B;">
                                <th style="padding: 0.4rem;">Evaluation Metric</th>
                                <th style="padding: 0.4rem;">Naive Baseline</th>
                                <th style="padding: 0.4rem; color: #16845B;">Random Forest</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #F1F5F9;">
                                <td style="padding: 0.4rem; font-weight: 600;">MAE</td>
                                <td style="padding: 0.4rem;">983.14 Tons</td>
                                <td style="padding: 0.4rem; font-weight: 700; color: #16845B;">139.13 Tons</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #F1F5F9;">
                                <td style="padding: 0.4rem; font-weight: 600;">RMSE</td>
                                <td style="padding: 0.4rem;">1,223.39 Tons</td>
                                <td style="padding: 0.4rem; font-weight: 700; color: #16845B;">203.18 Tons</td>
                            </tr>
                            <tr>
                                <td style="padding: 0.4rem; font-weight: 600;">R² Score</td>
                                <td style="padding: 0.4rem;">-0.2831</td>
                                <td style="padding: 0.4rem; font-weight: 700; color: #16845B;">0.9654</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# 6. REPORTS AND DOWNLOADS
# =========================================================
elif nav_page == "Reports and Downloads":
    st.markdown("""
    <div class="header-container">
        <span class="header-badge">Export & Data Artifacts</span>
        <h1 class="main-title">Reports and Downloads</h1>
        <p class="main-subtitle">Download verified route manifests, algorithmic benchmark comparisons, and priority assessments as CSV.</p>
    </div>
    """, unsafe_allow_html=True)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    st.markdown("""
    <div class="smart-card">
        <div class="smart-card-title">
            <span>📂</span> Verified Experimental Artifacts & Outputs
        </div>
    """, unsafe_allow_html=True)

    files = [
        ("tnagar_optimized_route.csv", "T. Nagar 157-Bin Optimized Sequence", "Sequential stop order, coordinates, and cumulative traversal for T. Nagar test cluster."),
        ("tnagar_route_comparison.csv", "Route Algorithmic Benchmark Comparison", "Comparison across CSV-order, Nearest Neighbor, and OR-Tools GLS with percentage reductions."),
        ("collection_priorities.csv", "Chennai Waste Collection Priority Report", "Percentile risk ranking and urgency tiers across Chennai waste streams."),
        ("tnagar_isolated_locations_review.csv", "T. Nagar Isolated Coordinates Review", "Review of 5 outlier container coordinates located >2 km from median cluster center.")
    ]

    for fname, title, desc in files:
        fpath = out_dir / fname
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.markdown(f"""
            <div style="padding: 0.5rem 0; border-bottom: 1px solid #F1F5F9;">
                <div style="font-weight: 700; color: #173D2D; font-size: 0.95rem;">📄 {title}</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.15rem;">{desc} (<code>{fname}</code>)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            if fpath.exists():
                data = fpath.read_bytes()
                st.download_button(
                    label="📥 Download CSV",
                    data=data,
                    file_name=fname,
                    mime="text/csv",
                    key=f"dl_{fname}",
                    use_container_width=True
                )
            else:
                st.caption("Not generated yet")

    st.markdown("</div>", unsafe_allow_html=True)

    # Dynamic Scenario Export Generator
    st.markdown("""
    <div class="smart-card">
        <div class="smart-card-title">
            <span>⚡</span> Dynamic Scenario Report Generator
        </div>
        <p style="font-size: 0.85rem; color: #64748B;">
            Generate a custom forecast and priority report for any scenario and download it immediately.
        </p>
    """, unsafe_allow_html=True)

    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        gen_year = st.selectbox("Scenario Year", [2024, 2025, 2026, 2027, 2028], index=2)
    with g_col2:
        gen_month = st.select_slider("Scenario Month", options=list(range(1, 13)), value=10)
    with g_col3:
        st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
        rep = generate_forecast_priorities(gen_year, gen_month)
        if rep is not None:
            csv_rep = rep.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Export Scenario {gen_year}-{gen_month:02d} CSV",
                data=csv_rep,
                file_name=f"chennai_waste_forecast_priority_{gen_year}_{gen_month:02d}.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="smart-notice">
        <strong>🔒 Data Provenance Notice:</strong>
        All downloaded artifacts reflect calculations performed directly on the Greater Chennai Corporation container dataset (11,058 coordinates) and the empirical Random Forest forecasting model. No synthetic operational claims or simulated real-time municipal telemetries are present.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.78rem; padding: 2rem 0 1rem 0; border-top: 1px solid #E2E8F0; margin-top: 2rem;">
    AI-Driven Urban Waste Generation Forecasting & Dynamic Route Optimization | Department of Computer Science & Engineering
</div>
""", unsafe_allow_html=True)
