# Real-Time Heart Rate Anomaly Detection

A real-time machine learning application that detects unusual heart-rate readings using an Isolation Forest model.

## Project Overview

This project monitors heart-rate data and identifies unusual values as potential anomalies. The backend provides API endpoints for health checks and real-time predictions, while the frontend provides a dashboard for displaying heart-rate activity and anomaly alerts.

## Features

- Real-time heart-rate anomaly detection
- Isolation Forest machine learning model
- FastAPI backend
- WebSocket support for real-time communication
- Interactive frontend dashboard
- Anomaly score for each prediction
- Health check API endpoint
- Adjustable anomaly detection workflow

## Technologies Used

- Python
- FastAPI
- Scikit-learn
- Isolation Forest
- WebSockets
- HTML
- CSS
- JavaScript
- Uvicorn

## Project Structure

```text
heart-rate-anomaly-detection/
│
├── backend/
│   ├── data_generator.py
│   ├── main.py
│   ├── model.py
│   └── ws_test.py
│
├── frontend/
│   └── index.html
│
└── models/
    └── heart_rate_isolation_forest.joblib
