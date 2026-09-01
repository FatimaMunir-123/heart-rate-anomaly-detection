import random
from datetime import datetime, timezone


def generate_normal_heart_rate():
    """Generate a synthetic normal heart-rate reading."""
    heart_rate = round(random.gauss(75, 5), 1)
    heart_rate = max(55, min(100, heart_rate))
    return heart_rate


def generate_anomaly_heart_rate():
    """Generate a synthetic abnormal heart-rate reading."""
    anomaly_type = random.choice(
        [
            "high_spike",
            "low_drop",
            "very_high",
            "very_low",
        ]
    )

    if anomaly_type == "high_spike":
        return round(random.uniform(120, 150), 1)

    if anomaly_type == "low_drop":
        return round(random.uniform(40, 50), 1)

    if anomaly_type == "very_high":
        return round(random.uniform(150, 180), 1)

    return round(random.uniform(30, 40), 1)


def generate_heart_rate(anomaly_probability=0.10):
    """
    Generate one synthetic heart-rate reading.

    anomaly_probability controls how often an abnormal
    reading is generated.
    """
    is_anomaly = random.random() < anomaly_probability

    if is_anomaly:
        heart_rate = generate_anomaly_heart_rate()
    else:
        heart_rate = generate_normal_heart_rate()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": heart_rate,
        "is_anomaly": is_anomaly,
    }


if __name__ == "__main__":
    for _ in range(30):
        reading = generate_heart_rate()
        print(reading)