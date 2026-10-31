
import streamlit as st
import requests
import json
import joblib
import pandas as pd
import numpy as np
import re

import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS = BASE_DIR / "artifacts"

import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="PredictOps AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. LOAD ARTIFACTS
# =========================================================

required_files = [
    "best_model.pkl",
    "feature_columns.pkl",
    "demo_engines.json",
    "demo_engine_history.json",
    "sensor_ranges.json"
]

missing_files = [
    file for file in required_files
    if not (ARTIFACTS / file).exists()
]

if missing_files:
    st.error(f"Missing artifacts: {missing_files}")
    st.stop()


@st.cache_resource
def load_model():

    return joblib.load(
        ARTIFACTS / "best_model.pkl"
    )


model = load_model()

feature_columns = joblib.load(
    ARTIFACTS / "feature_columns.pkl"
)

with open(ARTIFACTS / "demo_engines.json") as f:
    engines = json.load(f)

with open(ARTIFACTS / "demo_engine_history.json") as f:
    engine_history = json.load(f)

with open(ARTIFACTS / "sensor_ranges.json") as f:
    sensor_ranges = json.load(f)


# =========================================================
# 3. SENSOR DEFINITIONS
# =========================================================

sensor_names = {

    "s_1": "Fan inlet temperature (T2)",
    "s_2": "LPC outlet temperature (T24)",
    "s_3": "HPC outlet temperature (T30)",
    "s_4": "LPT outlet temperature (T50)",
    "s_5": "Fan inlet pressure (P2)",
    "s_6": "Bypass duct pressure (P15)",
    "s_7": "HPC outlet pressure (P30)",
    "s_8": "Physical fan speed (Nf)",
    "s_9": "Physical core speed (Nc)",
    "s_10": "Engine pressure ratio (epr)",
    "s_11": "HPC outlet static pressure (Ps30)",
    "s_12": "Fuel flow ratio (phi)",
    "s_13": "Corrected fan speed (NRf)",
    "s_14": "Corrected core speed (NRc)",
    "s_15": "Bypass ratio (BPR)",
    "s_16": "Burner fuel-air ratio (farB)",
    "s_17": "HPT coolant bleed (htBleed)",
    "s_18": "Demanded fan speed (Nf_dmd)",
    "s_19": "Corrected demanded fan speed (PCNfR_dmd)",
    "s_20": "HPT coolant bleed flow (W31)",
    "s_21": "LPT coolant bleed flow (W32)"
}


sensor_columns = [
    col for col in feature_columns
    if re.fullmatch(r"s_\d+", col)
]


# =========================================================
# 4. CUSTOM STYLING
# =========================================================

st.markdown("""
<style>
:root { --bg:#0b1220; --panel:#121e30; --border:#263950; --text:#e8eef7; --muted:#91a2b8; --accent:#39c6a0; }
.stApp { background:var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1680px; padding:0.7rem 1.25rem 1.1rem; }
section[data-testid="stSidebar"] { background:#101b2b; min-width:225px; max-width:225px; }
section[data-testid="stSidebar"] > div { padding-top:0.8rem; }
h1,h2,h3 { color:#f4f7fb !important; letter-spacing:-0.02em; }
h1 { font-size:1.65rem !important; margin:0 0 0.1rem 0 !important; }
h2 { font-size:1.2rem !important; margin:0.2rem 0 !important; }
h3 { font-size:1rem !important; margin:0.15rem 0 !important; }
p,label,.stMarkdown { color:#c4cfdd; }
[data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stVerticalBlock"] { gap:0.48rem; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#16263b,#111d2e); border:1px solid var(--border); padding:0.55rem 0.75rem; border-radius:9px; min-height:72px; }
div[data-testid="stMetricLabel"] { font-size:0.67rem; color:#9fb0c4; }
div[data-testid="stMetricValue"] { color:var(--text); font-size:1.35rem; }
div[data-testid="stForm"] { background:var(--panel); padding:0.65rem 0.75rem; border:1px solid var(--border); border-radius:10px; }
div[data-testid="stForm"] [data-testid="stVerticalBlock"] { gap:0.28rem; }
div.stButton > button,div[data-testid="stFormSubmitButton"] button { background:var(--accent); color:#071711; border:0; border-radius:7px; font-weight:700; min-height:2.25rem; }
div.stButton > button:hover,div[data-testid="stFormSubmitButton"] button:hover { background:#56d9b4; color:#071711; }
div[data-testid="stNumberInput"] input { background:#1b293b; color:var(--text); font-size:0.84rem; }
[data-testid="stAlert"] { padding:0.4rem 0.65rem; border-radius:8px; }
[data-testid="stPlotlyChart"] { margin:0 !important; }
hr { margin:0.4rem 0 !important; border-color:#263950; }
footer { visibility:hidden; }
@media(max-width:900px) { .block-container { padding:0.55rem 0.65rem 0.9rem; } section[data-testid="stSidebar"] { min-width:0; max-width:none; } }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:

    st.title("PREDICTOPS AI")

    st.caption("INTELLIGENT ENGINE MONITORING")

    st.divider()

    st.subheader("Control Center")

    engine_ids = sorted(
        [int(x) for x in engines.keys()]
    )

    selected_engine = st.selectbox(
        "Select Engine",
        engine_ids
    )

    st.divider()

    st.subheader("Model Information")

    st.write("Algorithm: XGBoost")
    st.write("Optimization: Optuna")
    st.write("Explainability: SHAP analysis")
    st.write(f"Input features: {len(feature_columns)}")

    st.divider()

    st.caption("Dataset: NASA C-MAPSS FD001")
    st.caption("Mode: Historical demonstration data")


# =========================================================
# 6. HEADER AND SYSTEM METRICS
# =========================================================

st.title("Engine Intelligence Center")

st.caption(
    "AI-powered turbofan health monitoring and "
    "remaining useful life prediction."
)

st.caption(
    "PREDICTIVE MAINTENANCE / TELEMETRY ANALYTICS / INFERENCE"
)

try:

    api_status = (
        requests.get(API_URL, timeout=2).status_code == 200
    )

except requests.exceptions.RequestException:

    api_status = False


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "MONITORED ENGINES",
        len(engine_ids)
    )

with c2:

    st.metric(
        "VALIDATION RMSE",
        "14.83 cycles"
    )

with c3:

    st.metric(
        "VALIDATION MAE",
        "10.33 cycles"
    )

with c4:

    st.metric(
        "API STATUS",
        "ONLINE" if api_status else "OFFLINE"
    )

st.divider()


# =========================================================
# 7. MAIN DASHBOARD LAYOUT
# =========================================================

left, right = st.columns([1.25, 1.0], gap="medium")

with left:
        st.subheader("Telemetry Input Console")

        st.caption(
            f"Editing measurements for Engine #{selected_engine}"
        )

        base_features = engines[str(selected_engine)]

        history_rows = engine_history[str(selected_engine)]

        edited_values = {}

        with st.form("sensor_input_form"):

            st.markdown("### Sensor Measurements")

            st.info(
                "Enter sensor readings. Reference intervals show "
                "the 1st–99th percentile of the training dataset. "
                "These are statistical references, not safety limits."
            )

            input_cols = st.columns(2)

            for i, sensor in enumerate(sensor_columns):

                label = sensor_names.get(
                    sensor,
                    "Sensor measurement"
                )

                value = float(base_features[sensor])

                ranges = sensor_ranges.get(sensor, {})

                low = ranges.get("p01")
                high = ranges.get("p99")

                with input_cols[i % 2]:

                    edited_values[sensor] = st.number_input(

                        f"{sensor} | {label}",

                        value=value,

                        format="%.5f",

                        help=(
                            f"Training P1-P99: "
                            f"{low:.4f} to {high:.4f}. "
                            "Reference only."
                            if low is not None and high is not None
                            else "No reference range available."
                        ),

                        key=f"sensor_{selected_engine}_{sensor}"

                    )

                    if low is not None and high is not None:

                        st.caption(
                            f"Reference: {low:.3f} - {high:.3f}"
                        )

            st.divider()

            submitted = st.form_submit_button(
                "RUN ENGINE HEALTH PREDICTION",
                use_container_width=True
            )

with right:
        st.subheader("Turbofan Engine Architecture")

        st.caption(
            "Interactive conceptual engineering schematic"
        )

        fig_engine = go.Figure()

        # Engine casing
        fig_engine.add_shape(
            type="rect",
            x0=0.05,
            x1=0.95,
            y0=0.34,
            y1=0.68,
            fillcolor="#172b43",
            line=dict(
                color="#64748b",
                width=2
            )
        )

        # Engine components
        components = [

            (0.08, 0.20, "FAN", "#2563eb"),
            (0.22, 0.31, "LPC", "#3b82f6"),
            (0.34, 0.48, "HPC", "#6366f1"),
            (0.51, 0.65, "COMBUSTOR", "#d97706"),
            (0.68, 0.79, "HPT", "#059669"),
            (0.80, 0.90, "LPT", "#0d9488")

        ]

        for x0, x1, name, color in components:

            fig_engine.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=0.40,
                y1=0.62,
                fillcolor=color,
                opacity=0.9,
                line=dict(
                    color="white",
                    width=1
                )
            )

            fig_engine.add_annotation(
                x=(x0 + x1) / 2,
                y=0.51,
                text=f"<b>{name}</b>",
                showarrow=False,
                font=dict(
                    size=10,
                    color="white"
                )
            )

        # Fan blades
        for angle in range(0, 360, 45):

            radians = np.radians(angle)

            x_end = 0.14 + 0.055 * np.cos(radians)
            y_end = 0.51 + 0.09 * np.sin(radians)

            fig_engine.add_shape(
                type="line",
                x0=0.14,
                y0=0.51,
                x1=x_end,
                y1=y_end,
                line=dict(
                    color="#bfdbfe",
                    width=3
                )
            )

        # Exhaust nozzle
        fig_engine.add_shape(
            type="path",
            path="M 0.90,0.40 L 0.97,0.45 L 0.97,0.57 L 0.90,0.62 Z",
            fillcolor="#475569",
            line=dict(
                color="#94a3b8",
                width=2
            )
        )

        # Airflow
        fig_engine.add_annotation(
            x=0.92,
            y=0.77,
            ax=0.08,
            ay=0.77,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="AIRFLOW DIRECTION",
            showarrow=True,
            arrowhead=3,
            arrowcolor="#34d399",
            font=dict(
                color="#34d399",
                size=11
            )
        )

        # Sensor measurement locations
        sensor_points = [

            (0.27, 0.68, "s_2 | T24", 0.22, 0.91),
            (0.43, 0.68, "s_3 | T30", 0.42, 0.96),
            (0.86, 0.68, "s_4 | T50", 0.78, 0.91),
            (0.46, 0.34, "s_7 | P30", 0.32, 0.12),
            (0.45, 0.34, "s_11 | Ps30", 0.48, 0.04),
            (0.58, 0.68, "s_12 | phi", 0.59, 0.94),
            (0.84, 0.34, "s_15 | BPR", 0.83, 0.12)

        ]

        for x, y, label, lx, ly in sensor_points:

            fig_engine.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="#fbbf24",
                        line=dict(
                            color="white",
                            width=1
                        )
                    ),
                    hovertemplate=label + "<extra></extra>",
                    showlegend=False
                )
            )

            fig_engine.add_annotation(
                x=x,
                y=y,
                ax=lx,
                ay=ly,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text=f"<b>{label}</b>",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#fbbf24",
                font=dict(
                    size=10,
                    color="#fbbf24"
                ),
                bgcolor="#172b43",
                bordercolor="#475569",
                borderpad=4
            )

        fig_engine.update_layout(

            template="plotly_dark",

            height=350,

            margin=dict(
                l=15,
                r=15,
                t=20,
                b=20
            ),

            paper_bgcolor="#111c2e",
            plot_bgcolor="#111c2e",

            showlegend=False,

            xaxis=dict(
                range=[0, 1.02],
                visible=False,
                fixedrange=True
            ),

            yaxis=dict(
                range=[0, 1.02],
                visible=False,
                fixedrange=True,
                scaleanchor="x"
            )
        )

        st.plotly_chart(
            fig_engine,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

        st.caption(
            "Conceptual engine layout. Sensor locations are "
            "approximate measurement stations, not exact "
            "hardware positions."
        )

# =========================================================
# 8. ROLLING FEATURE ENGINEERING
# =========================================================

def update_rolling_features(
    features,
    history,
    edited
):

    features = features.copy()

    # Detect engineered features from actual model schema
    rolling_pattern = re.compile(
        r"^(s_\d+)_(mean|std|trend)_5$"
    )

    rolling_sensors = {}

    for col in feature_columns:

        match = rolling_pattern.fullmatch(col)

        if match:

            sensor = match.group(1)
            statistic = match.group(2)

            rolling_sensors.setdefault(
                sensor, set()
            ).add(statistic)

    for sensor, statistics in rolling_sensors.items():

        if sensor not in edited:
            continue

        if not history:
            continue

        # Last five observations including current cycle
        previous_values = [

            float(row[sensor])

            for row in history[-5:]

            if sensor in row

        ]

        if not previous_values:
            continue

        # Replace current reading with user input
        previous_values[-1] = float(
            edited[sensor]
        )

        values = pd.Series(previous_values)

        # Mean
        if "mean" in statistics:

            features[f"{sensor}_mean_5"] = float(
                values.mean()
            )

        # Standard deviation
        if "std" in statistics:

            features[f"{sensor}_std_5"] = (

                float(values.std(ddof=1))

                if len(values) > 1

                else 0.0

            )

        # Trend: difference from five cycles earlier
        if "trend" in statistics:

            if (
                len(history) >= 6
                and sensor in history[-6]
            ):

                features[f"{sensor}_trend_5"] = (

                    float(edited[sensor])

                    - float(history[-6][sensor])

                )

            else:

                features[f"{sensor}_trend_5"] = 0.0

    return features


# =========================================================
# 10. PREDICTION API
# =========================================================

if submitted:

    try:

        updated_features = update_rolling_features(

            base_features,

            history_rows,

            edited_values

        )

        missing = [

            col for col in feature_columns

            if col not in updated_features

        ]

        if missing:

            st.error(
                f"Missing model features: {missing}"
            )

        else:

            payload = {

                "features": {

                    col: float(updated_features[col])

                    for col in feature_columns

                }

            }

            with st.spinner(
                "Analyzing engine telemetry..."
            ):

                response = requests.post(

                    f"{API_URL}/predict",

                    json=payload,

                    timeout=30

                )

            response.raise_for_status()

            result = response.json()

            st.session_state["prediction"] = result

            st.session_state["prediction_engine"] = (
                selected_engine
            )

            st.session_state["edited_values"] = (
                edited_values.copy()
            )

    except requests.exceptions.RequestException as e:

        st.error(
            f"Prediction API error: {e}"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


with right:
    st.divider()

    st.subheader("Remaining Useful Life Analytics")

    if (

        "prediction" in st.session_state

        and st.session_state.get(
            "prediction_engine"
        ) == selected_engine

    ):

        result = st.session_state["prediction"]

        rul = float(result["predicted_rul"])

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "PREDICTED RUL",
                f"{rul:.2f} cycles"
            )

        with r2:

            st.metric(
                "ENGINE ID",
                selected_engine
            )

        with r3:

            st.metric(
                "PREDICTION STATUS",
                "Complete"
            )

        st.success(
            "Prediction generated successfully through FastAPI."
        )

        st.info(
            "This is a model estimate, not a certified "
            "airworthiness or safety assessment. Maintenance "
            "decisions require an approved operational policy."
        )

        # Empirical sensor reference warnings
        edited = st.session_state.get(
            "edited_values", {}
        )

        out_of_range = []

        for sensor, value in edited.items():

            ranges = sensor_ranges.get(sensor, {})

            low = ranges.get("p01")
            high = ranges.get("p99")

            if low is not None and high is not None:

                if value < low or value > high:

                    out_of_range.append(
                        (sensor, value, low, high)
                    )

        if out_of_range:

            st.warning(
                "Some readings fall outside the training "
                "dataset's P1-P99 reference intervals."
            )

            for sensor, value, low, high in out_of_range:

                st.write(
                    f"**{sensor}**: {value:.4f} "
                    f"(Reference: {low:.4f} - {high:.4f})"
                )

        else:

            st.success(
                "All edited readings fall within their "
                "training-data reference intervals."
            )

    else:

        st.info(
            "Select an engine, edit sensor measurements "
            "and run a prediction to view results."
        )

# =========================================================
# 12. HISTORICAL SENSOR TRENDS
# =========================================================

st.divider()

st.subheader("Historical Sensor Trends")

history_df = pd.DataFrame(history_rows)

trend_sensors = [

    s for s in ["s_4", "s_11", "s_12", "s_15"]

    if s in history_df.columns

]

if not history_df.empty and trend_sensors:

    trend_df = history_df[

        ["time_cycles"] + trend_sensors

    ].melt(

        id_vars="time_cycles",

        var_name="Sensor",

        value_name="Reading"

    )

    trend_df["Measurement"] = (
        trend_df["Sensor"].map(sensor_names)
    )

    fig_trend = px.line(

        trend_df,

        x="time_cycles",

        y="Reading",

        color="Measurement",

        title="Engine Sensor Behaviour Across Operating Cycles"

    )

    fig_trend.update_layout(

        template="plotly_dark",

        paper_bgcolor="#0b1120",

        plot_bgcolor="#111c2e",

        height=430,

        xaxis_title="Operating Cycles",

        yaxis_title="Sensor Reading",

        legend_title="Physical Measurement"

    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True
    )

    st.caption(
        "Historical readings from the selected engine. "
        "Different sensors use different measurement scales."
    )


# =========================================================
# 13. MODEL FEATURE IMPORTANCE
# =========================================================

st.divider()

st.subheader("Model Intelligence")

if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    ).head(12)

    fig_importance = px.bar(

        importance.sort_values("Importance"),

        x="Importance",

        y="Feature",

        orientation="h",

        title="Top 12 Model Features"

    )

    fig_importance.update_layout(

        template="plotly_dark",

        paper_bgcolor="#0b1120",

        plot_bgcolor="#111c2e",

        height=430,

        xaxis_title="Feature Importance",

        yaxis_title="Feature"

    )

    st.plotly_chart(

        fig_importance,

        use_container_width=True

    )

else:

    st.info(
        "Feature importance is unavailable for this model."
    )


# =========================================================
# 14. FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#64748b;
    padding:20px;
">

    <h3 style="color:#34d399;">
        PREDICTOPS AI
    </h3>

    Intelligent Predictive Maintenance Platform

    <br><br>

    XGBoost · Optuna · SHAP · FastAPI · Streamlit

    <br><br>

    NASA C-MAPSS Turbofan Degradation Dataset

</div>
""", unsafe_allow_html=True)
