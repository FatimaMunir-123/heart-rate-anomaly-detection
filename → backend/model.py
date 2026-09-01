from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_rate_isolation_forest.joblib"


def train_model() -> IsolationForest:
    """
    Train an Isolation Forest using synthetic normal heart-rate data.
    """

    rng = np.random.default_rng(42)

    # Synthetic normal resting heart-rate training data.
    heart_rates = rng.normal(loc=75, scale=5, size=1000)

    # Keep training data within our defined normal test range.
    heart_rates = np.clip(heart_rates, 55, 100)

    X = heart_rates.reshape(-1, 1)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )

    model.fit(X)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


def load_model() -> IsolationForest:
    """Load the trained Isolation Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run train_model() first."
        )

    return joblib.load(MODEL_PATH)


def predict_heart_rate(
    model: IsolationForest,
    heart_rate: float,
) -> dict:
    """Predict whether a heart-rate reading is anomalous."""

    X = np.array([[heart_rate]])

    prediction = model.predict(X)[0]
    score = model.decision_function(X)[0]

    return {
        "heart_rate": heart_rate,
        "prediction": "anomaly" if prediction == -1 else "normal",
        "anomaly_score": round(float(score), 4),
    }


if __name__ == "__main__":
    model = train_model()

    test_values = [70, 75, 82, 140, 45, 165]

    for value in test_values:
        result = predict_heart_rate(model, value)
        print(result)

    print(f"\nModel saved to: {MODEL_PATH}")
