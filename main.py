from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import random
import asyncio
from datetime import datetime, timezone


app = FastAPI(
    title="Real-Time Heart Rate Anomaly Detection",
    version="0.1.0",
)


# Allow the frontend to communicate with the FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained Isolation Forest model
model = load("models/heart_rate_isolation_forest.joblib")


# Anomaly sensitivity threshold
threshold = 0.0


class HeartRateRequest(BaseModel):
    heart_rate: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "heart-rate-anomaly-detection",
    }


@app.post("/predict")
def predict(data: HeartRateRequest):

    score = float(
        model.decision_function(
            [[data.heart_rate]]
        )[0]
    )

    prediction = (
        "anomaly"
        if score < threshold
        else "normal"
    )

    return {
        "heart_rate": data.heart_rate,
        "prediction": prediction,
        "anomaly_score": round(score, 4),
    }


@app.post("/threshold")
def update_threshold(value: float):

    global threshold

    threshold = value

    return {
        "threshold": threshold,
        "status": "updated",
    }


@app.get("/threshold")
def get_threshold():

    return {
        "threshold": threshold,
    }


@app.websocket("/ws/heart-rate")
async def heart_rate_websocket(
    websocket: WebSocket
):

    await websocket.accept()

    try:

        while True:

            # Generate normal heart-rate readings
            heart_rate = round(
                random.gauss(72, 6),
                1
            )


            # Occasionally generate abnormal readings
            if random.random() < 0.08:

                heart_rate = round(
                    random.choice([
                        random.uniform(40, 48),
                        random.uniform(130, 150)
                    ]),
                    1
                )


            # Calculate anomaly score
            score = float(
                model.decision_function(
                    [[heart_rate]]
                )[0]
            )


            # Apply current threshold
            prediction = (
                "anomaly"
                if score < threshold
                else "normal"
            )


            result = {
                "heart_rate": heart_rate,
                "prediction": prediction,
                "anomaly_score": round(
                    score,
                    4
                ),
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }


            # Send live result to frontend
            await websocket.send_json(
                result
            )


            # Wait one second
            await asyncio.sleep(1)


    except Exception:
        pass