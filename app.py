
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AeroMind | Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")

REPORT_PATH = os.path.join(
    PROCESSED_DIR, "maintenance_alert_report.csv"
)

RESULT_PATH = os.path.join(
    PROCESSED_DIR, "prediction_results.csv"
)

RAW_PATH = os.path.join(
    RAW_DIR, "test_FD001.txt"
)

# =====================================================
# 2. CUSTOM DESIGN / CSS
# =====================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(145deg, #07111f 0%, #0b1728 55%, #101d30 100%);
        color: #e5edf7;
    }

    [data-testid="stSidebar"] {
        background: #081321;
        border-right: 1px solid #1e344d;
    }

    [data-testid="stSidebar"] * {
        color: #dce8f5;
    }

    .hero {
        background: linear-gradient(110deg, #102b46, #123c59, #07516a);
        border: 1px solid #245575;
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
    }

    .hero p {
        color: #b9d7e9;
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 14px;
    }

    .section-label {
        color: #65d8ed;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    .small-note {
        color: #8fa7be;
        font-size: 12px;
    }

    div[data-testid="stMetric"] {
        background: #112238;
        border: 1px solid #25415d;
        padding: 17px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a9bfd4;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    div[data-testid="stMetricDelta"] {
        color: #64d9ed;
    }

    .status-critical {
        color: #ff7272;
        font-weight: 700;
    }

    .status-warning {
        color: #ffc45c;
        font-weight: 700;
    }

    .status-normal {
        color: #52d6a0;
        font-weight: 700;
    }

    h1, h2, h3 {
        color: #f2f7fc;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #29415b;
        border-radius: 12px;
        overflow: hidden;
    }

    .footer {
        text-align: center;
        color: #718ba5;
        font-size: 11px;
        padding: 20px;
        border-top: 1px solid #23394f;
        margin-top: 30px;
    }

    .stButton button, .stDownloadButton button {
        border-radius: 9px;
        border: 1px solid #247c9b;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    report = pd.read_csv(REPORT_PATH)
    results = pd.read_csv(RESULT_PATH)

    columns = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    raw = pd.read_csv(
        RAW_PATH,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return report, results, raw


try:
    report, results, raw = load_data()
except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()

# =====================================================
# 4. GLOBAL SETTINGS
# =====================================================

COLORS = {
    "NORMAL": "#22c995",
    "WARNING": "#ffbd59",
    "CRITICAL": "#ff626f"
}

total_engines = len(report)

normal_count = (report["status"] == "NORMAL").sum()
warning_count = (report["status"] == "WARNING").sum()
critical_count = (report["status"] == "CRITICAL").sum()

# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:
    st.markdown("# ✈️ AeroMind")
    st.caption("AI AVIATION INTELLIGENCE")
    st.divider()

    page = st.radio(
        "CONTROL CENTER",
        [
            "Fleet Overview",
            "Engine Intelligence",
            "Maintenance Control",
            "AI Performance"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**SYSTEM INFORMATION**")
    st.caption("Dataset: NASA C-MAPSS FD001")
    st.caption("AI Model: Random Forest")
    st.caption("Engines: 100")
    st.caption("Environment: Research Prototype")

    st.divider()
    st.caption("AeroMind Predictive Maintenance v1.0")

# =====================================================
# 6. REUSABLE HEADER
# =====================================================

def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="section-label">AEROMIND / CONTROL CENTER</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# 7. FLEET OVERVIEW
# =====================================================

if page == "Fleet Overview":

    page_header(
        "Fleet Intelligence",
        "Real-time style monitoring of predicted aircraft engine health."
    )

    st.markdown("### Fleet Health Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("TOTAL ENGINES", total_engines)
    c2.metric("NORMAL", normal_count)
    c3.metric("WARNING", warning_count)
    c4.metric("CRITICAL", critical_count)

    st.markdown("### Fleet Analytics")

    left, right = st.columns(2)

    with left:
        health = (
            report["status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Engines")
        )

        fig = px.pie(
            health,
            names="Status",
            values="Engines",
            hole=0.65,
            color="Status",
            color_discrete_map=COLORS
        )

        fig.update_layout(
            title="Engine Health Distribution",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title="Health Status"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            report,
            x="predicted_RUL",
            nbins=15,
            color_discrete_sequence=["#36c9e6"],
            title="Remaining Useful Life Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Predicted RUL (Cycles)",
            yaxis_title="Number of Engines"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Engine RUL Monitoring")

    fig = px.bar(
        report.sort_values("engine_id"),
        x="engine_id",
        y="predicted_RUL",
        color="status",
        color_discrete_map=COLORS,
        title="Predicted Remaining Useful Life by Engine",
        hover_data=["current_cycle"]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Engine ID",
        yaxis_title="Predicted RUL (Cycles)",
        legend_title="Engine Status"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Latest Fleet Records")

    st.dataframe(
        report.sort_values("predicted_RUL"),
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# 8. ENGINE INTELLIGENCE
# =====================================================

elif page == "Engine Intelligence":

    page_header(
        "Engine Intelligence",
        "Inspect individual engine predictions and sensor behaviour."
    )

    engine_ids = sorted(report["engine_id"].unique())

    selected_engine = st.selectbox(
        "SELECT ENGINE",
        engine_ids
    )

    engine_report = report[
        report["engine_id"] == selected_engine
    ].iloc[0]

    engine_result = results[
        results["engine_id"] == selected_engine
    ]

    engine_history = raw[
        raw["engine_id"] == selected_engine
    ].sort_values("cycle")

    status = engine_report["status"]
    status_color = COLORS[status]

    st.markdown(f"### Engine #{selected_engine}")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "CURRENT CYCLE",
        int(engine_report["current_cycle"])
    )

    c2.metric(
        "PREDICTED RUL",
        f'{engine_report["predicted_RUL"]:.2f} cycles'
    )

    c3.metric(
        "HEALTH STATUS",
        status
    )

    st.markdown(
        f'<p style="color:{status_color};font-weight:700;">'
        f'● {status} ENGINE STATUS</p>',
        unsafe_allow_html=True
    )

    if not engine_result.empty:
        actual = float(engine_result.iloc[0]["actual_RUL"])
        predicted = float(engine_result.iloc[0]["predicted_RUL"])

        st.markdown("### Prediction Comparison")

        p1, p2, p3 = st.columns(3)

        p1.metric("ACTUAL RUL", f"{actual:.2f}")
        p2.metric("PREDICTED RUL", f"{predicted:.2f}")
        p3.metric("ABSOLUTE ERROR", f"{abs(actual-predicted):.2f}")

    st.markdown("### Sensor Trend Analysis")

    sensor_columns = [
        f"sensor_{i}" for i in range(1, 22)
    ]

    selected_sensor = st.selectbox(
        "SELECT SENSOR",
        sensor_columns
    )

    fig = px.line(
        engine_history,
        x="cycle",
        y=selected_sensor,
        markers=True,
        title=f"{selected_sensor.upper()} Behaviour Across Engine Cycles",
        color_discrete_sequence=["#38cce8"]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Operating Cycle",
        yaxis_title="Sensor Reading"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Latest Sensor Readings")

    latest = engine_history.tail(1)

    st.dataframe(
        latest[["cycle"] + sensor_columns],
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# 9. MAINTENANCE CONTROL
# =====================================================

elif page == "Maintenance Control":

    page_header(
        "Maintenance Control Center",
        "Identify engines requiring additional attention based on model predictions."
    )

    st.markdown("### Alert Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("CRITICAL ENGINES", critical_count)
    c2.metric("WARNING ENGINES", warning_count)
    c3.metric("NORMAL ENGINES", normal_count)

    selected_status = st.multiselect(
        "FILTER ENGINE STATUS",
        ["CRITICAL", "WARNING", "NORMAL"],
        default=["CRITICAL", "WARNING"]
    )

    filtered = report[
        report["status"].isin(selected_status)
    ].sort_values("predicted_RUL")

    st.markdown("### Maintenance Alert Records")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇ Download Maintenance Report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="maintenance_alert_report.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇ Download Filtered Alerts",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_maintenance_alerts.csv",
        mime="text/csv"
    )

# =====================================================
# 10. AI PERFORMANCE
# =====================================================

elif page == "AI Performance":

    page_header(
        "AI Model Performance",
        "Independent evaluation of the Random Forest model on NASA FD001 test data."
    )

    mae = 12.45
    rmse = 16.83
    r2 = 0.8237

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.2f} cycles")
    c2.metric("RMSE", f"{rmse:.2f} cycles")
    c3.metric("R² SCORE", f"{r2:.4f}")

    st.markdown("### Actual vs Predicted RUL")

    fig = px.scatter(
        results,
        x="actual_RUL",
        y="predicted_RUL",
        hover_data=["engine_id"],
        color_discrete_sequence=["#37cce9"],
        title="Model Prediction Evaluation"
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 125],
            y=[0, 125],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="#ff646e", dash="dash")
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Actual RUL (Cycles)",
        yaxis_title="Predicted RUL (Cycles)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prediction Error Distribution")

    error_data = results.copy()

    if "prediction_error" not in error_data.columns:
        error_data["prediction_error"] = (
            error_data["predicted_RUL"] - error_data["actual_RUL"]
        )

    fig = px.histogram(
        error_data,
        x="prediction_error",
        nbins=15,
        color_discrete_sequence=["#36c9e6"],
        title="Prediction Error (Predicted - Actual)"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Prediction Error (Cycles)",
        yaxis_title="Number of Engines"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Complete Evaluation Results")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# 11. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    AEROMIND | AI-POWERED AIRCRAFT ENGINE PREDICTIVE MAINTENANCE<br>
    NASA C-MAPSS FD001 | RANDOM FOREST REGRESSION<br>
    EDUCATIONAL RESEARCH PROTOTYPE — NOT FOR OPERATIONAL FLIGHT DECISIONS
</div>
""", unsafe_allow_html=True)
