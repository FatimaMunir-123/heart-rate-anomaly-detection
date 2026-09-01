
import time
import random
import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Heart Rate Anomaly Detection",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Real-Time Heart Rate Anomaly Detection")
st.caption("Isolation Forest-based monitoring dashboard")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/heart_rate_isolation_forest.joblib")

model = load_model()

# Sidebar controls
st.sidebar.header("Detection Settings")

threshold = st.sidebar.slider(
    "Anomaly Sensitivity",
    min_value=0.01,
    max_value=1.00,
    value=0.50,
    step=0.01
)

refresh_rate = st.sidebar.slider(
    "Refresh Rate (seconds)",
    min_value=0.2,
    max_value=2.0,
    value=0.8,
    step=0.1
)

start = st.sidebar.button("Start Monitoring")

# Dashboard placeholders
metric_col1, metric_col2, metric_col3 = st.columns(3)

heart_rate_metric = metric_col1.empty()
status_metric = metric_col2.empty()
anomaly_metric = metric_col3.empty()

chart_placeholder = st.empty()
alert_placeholder = st.empty()

if start:
    heart_rates = []
    anomaly_points = []

    for i in range(100):
        # Simulated heart-rate data
        if random.random() < 0.08:
            heart_rate = random.choice([
                random.randint(35, 50),
                random.randint(120, 160)
            ])
        else:
            heart_rate = random.randint(65, 95)

        # Model prediction
        prediction = model.predict(
            np.array([[heart_rate]])
        )[0]

        is_anomaly = prediction == -1

        heart_rates.append(heart_rate)

        if is_anomaly:
            anomaly_points.append(heart_rate)
        else:
            anomaly_points.append(None)

        # Metrics
        heart_rate_metric.metric(
            "Current Heart Rate",
            f"{heart_rate} BPM"
        )

        if is_anomaly:
            status_metric.error("⚠️ ANOMALY DETECTED")
            alert_placeholder.warning(
                f"⚠️ Abnormal heart rate detected: {heart_rate} BPM"
            )
        else:
            status_metric.success("✅ NORMAL")
            alert_placeholder.info(
                "No anomaly detected."
            )

        anomaly_metric.metric(
            "Anomalies Detected",
            sum(x is not None for x in anomaly_points)
        )

        # Chart data
        chart_data = {
            "Heart Rate": heart_rates
        }

        chart_placeholder.line_chart(chart_data)

        time.sleep(refresh_rate)

else:
    heart_rate_metric.metric("Current Heart Rate", "-- BPM")
    status_metric.info("Monitoring stopped")
    anomaly_metric.metric("Anomalies Detected", "0")

    st.info("Click **Start Monitoring** from the sidebar to begin.")
