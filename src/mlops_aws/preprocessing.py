from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "target"


def create_dataset(test_size: float = 0.2, random_state: int = 42):
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    return train_df, test_df


def save_datasets(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    output_dir: str,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
