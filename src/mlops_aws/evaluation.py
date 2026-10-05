import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mlops_aws.preprocessing import TARGET_COLUMN


def evaluate_model(model, test_df: pd.DataFrame) -> dict:
    X = test_df.drop(columns=[TARGET_COLUMN])
    y = test_df[TARGET_COLUMN]

    predictions = model.predict(X)

    return {
        "classification_metrics": {
            "accuracy": {
                "value": accuracy_score(y, predictions),
            },
            "precision": {
                "value": precision_score(y, predictions),
            },
            "recall": {
                "value": recall_score(y, predictions),
            },
            "f1": {
                "value": f1_score(y, predictions),
            }
        }
    }


def save_metrics(metrics: dict, output_file: str):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as file:
        json.dump(metrics, file, indent=4)