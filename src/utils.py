import json
import os

import joblib


def save_results(models, metrics=None, results=None):
    payload = metrics if metrics is not None else results
    if payload is None:
        raise ValueError("Provide either 'metrics' or 'results'.")

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, f"outputs/models/{name}.pkl")

    with open("outputs/metrics/results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
