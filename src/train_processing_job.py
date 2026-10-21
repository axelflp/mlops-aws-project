import argparse
import tarfile
from pathlib import Path

import pandas as pd

from mlops_aws.training import save_model, train_model

INPUT_FILE = Path("/opt/ml/processing/input/train.csv")
MODEL_DIR = Path("/opt/ml/processing/model")
OUTPUT_DIR = Path("/opt/ml/processing/output")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    train_df = pd.read_csv(INPUT_FILE)

    model = train_model(
        train_df,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    save_model(
        model=model,
        model_dir=str(MODEL_DIR),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_archive = OUTPUT_DIR/"model.tar.gz"

    with tarfile.open(model_archive, "w:gz") as tar:
        tar.add(
            MODEL_DIR/"model.joblib",
            arcname="model.joblib",
        )

    print(f"Training rows: {len(train_df)}")
    print(f"Model artifact: {model_archive}")


if __name__ == '__main__':
    main()