import time
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Rate Anomaly Detection",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Real-Time Heart Rate Anomaly Detection")
st.caption("Live anomaly monitoring powered by Isolation Forest")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/heart_rate_isolation_forest.joblib")


model = load_model()


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Detection Settings")

sensitivity = st.sidebar.slider(
    "Anomaly Sensitivity",
    min_value=0.01,
    max_value=1.00,
    value=0.50,
    step=0.01,
    help="Higher sensitivity makes the detector more likely to flag unusual heart rates."
)

refresh_rate = st.sidebar.slider(
    "Refresh Rate (seconds)",
    min_value=0.2,
    max_value=2.0,
    value=0.8,
    step=0.1
)

max_points = st.sidebar.slider(
    "Chart History",
    min_value=20,
    max_value=100,
    value=50,
    step=10
)

start = st.sidebar.button("▶ Start Monitoring")


# -----------------------------
# Session state
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "heart_rates" not in st.session_state:
    st.session_state.heart_rates = []

if "scores" not in st.session_state:
    st.session_state.scores = []

if "anomalies" not in st.session_state:
    st.session_state.anomalies = []

if "alerts" not in st.session_state:
    st.session_state.alerts = []


if start:
    st.session_state.running = True


# -----------------------------
# Dashboard placeholders
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

heart_rate_metric = col1.empty()
status_metric = col2.empty()
anomaly_metric = col3.empty()
sensitivity_metric = col4.empty()

st.subheader("Live Heart Rate")

chart_placeholder = st.empty()

st.subheader("🚨 Anomaly Alerts")

alert_placeholder = st.empty()


# -----------------------------
# Monitoring
# -----------------------------
if st.session_state.running:

    for _ in range(1000):

        # Generate mostly normal heart rates
        if random.random() < 0.10:
            heart_rate = random.choice([
                random.randint(35, 50),
                random.randint(120, 160)
            ])
        else:
            heart_rate = random.randint(60, 100)

        # Model score
        score = float(model.decision_function(
            np.array([[heart_rate]])
        )[0])

        # Convert sensitivity into a dynamic threshold.
        #
        # Lower sensitivity = stricter detection
        # Higher sensitivity = easier anomaly detection
        threshold = 0.15 - (sensitivity * 0.30)

        is_anomaly = score < threshold

        # Store data
        st.session_state.heart_rates.append(heart_rate)
        st.session_state.scores.append(score)

        if is_anomaly:
            st.session_state.anomalies.append(heart_rate)

            timestamp = time.strftime("%H:%M:%S")

            st.session_state.alerts.insert(
                0,
                f"⚠️ **{timestamp}** — Anomaly detected: **{heart_rate} BPM**"
            )

        # Keep history limited
        st.session_state.heart_rates = (
            st.session_state.heart_rates[-max_points:]
        )

        st.session_state.scores = (
            st.session_state.scores[-max_points:]
        )

        # Metrics
        heart_rate_metric.metric(
            "Current Heart Rate",
            f"{heart_rate} BPM"
        )

        if is_anomaly:
            status_metric.error("⚠️ ANOMALY")
        else:
            status_metric.success("✅ NORMAL")

        anomaly_metric.metric(
            "Total Anomalies",
            len(st.session_state.anomalies)
        )

        sensitivity_metric.metric(
            "Sensitivity",
            f"{sensitivity:.2f}"
        )

        # Live chart
        chart_df = pd.DataFrame({
            "Heart Rate (BPM)": st.session_state.heart_rates
        })

        chart_placeholder.line_chart(
            chart_df,
            height=350
        )

        # Alerts
        if st.session_state.alerts:
            alert_placeholder.markdown(
                "\n\n".join(
                    st.session_state.alerts[:8]
                )
            )
        else:
            alert_placeholder.info(
                "No anomalies detected yet."
            )

        time.sleep(refresh_rate)

else:

    heart_rate_metric.metric(
        "Current Heart Rate",
        "-- BPM"
    )

    status_metric.info(
        "Monitoring stopped"
    )

    anomaly_metric.metric(
        "Total Anomalies",
        len(st.session_state.anomalies)
    )

    sensitivity_metric.metric(
        "Sensitivity",
        f"{sensitivity:.2f}"
    )

    chart_placeholder.info(
        "Click **▶ Start Monitoring** to begin live detection."
    )

    alert_placeholder.info(
        "No active alerts."
    )
