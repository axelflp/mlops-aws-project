from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_aws.preprocessing import TARGET_COLUMN


def train_model(train_df: pd.DataFrame, 
                max_iter: int = 1000,
                random_state: int =40,
) -> Pipeline:
    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", 
            LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                )
            ),

        ]
    )

    model.fit(X, y)

    return model


def save_model(model: Pipeline, model_dir: str):
    output_path = Path(model_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, output_path / "model.joblib")