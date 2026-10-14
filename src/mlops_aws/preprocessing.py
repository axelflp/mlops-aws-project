from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "target"

def split_dataset(
        dataframe: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
):
        if TARGET_COLUMN not in dataframe.columns:
             raise ValueError(f"Missing target column: {TARGET_COLUMN}")

        train_df, test_df = train_test_split(
             dataframe,
             test_size=test_size,
             random_state=random_state,
             stratify=dataframe[TARGET_COLUMN]
        )

        return (
             train_df.reset_index(drop=True),
             test_df.reset_index(drop=True),
        )


def create_dataset(test_size: float = 0.2, random_state: int = 42):
    dataset = load_breast_cancer(as_frame=True)

    return split_dataset(
         dataset.frame,
         test_size=test_size,
         random_state=random_state,
    )


def save_dataset(
          dataframe:pd.DataFrame,
          output_file: str,
):
     output_path = Path(output_file)
     output_path.parent.mkdir(parents=True, exist_ok=True)

     dataframe.to_csv(output_path, index=False)


def save_datasets(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    output_dir: str,
):
    output_path = Path(output_dir)
    save_dataset(train_df, output_path/"train.csv")
    save_dataset(test_df, output_path/"test.csv")
