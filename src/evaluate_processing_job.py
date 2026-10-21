import shutil
import tarfile
from pathlib import Path

import joblib
import pandas as pd

from mlops_aws.evaluation import evaluate_model, save_metrics

MODEL_ARCHIVE = Path("/opt/ml/processing/model/model.tar.gz")
MODEL_FILE = Path("/opt/ml/processing/model/model.joblib")
TEST_FILE = Path("/opt/ml/processing/test/test.csv")
OUTPUT_FILE = Path("/opt/ml/processing/evaluation/evaluation.json")


def extract_model():
    with tarfile.open(MODEL_ARCHIVE, "r:gz") as archive:
        member = archive.getmember("model.joblib")

        source = archive.extractfile(member)

        if source is None:
            raise RuntimeError("model.joblib not found in model.tar.gz")

        with MODEL_FILE.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def main():
    extract_model()

    model = joblib.load(MODEL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    metrics = evaluate_model(
        model=model,
        test_df=test_df,
    )

    save_metrics(
        metrics=metrics,
        output_file=str(OUTPUT_FILE),
    )

    f1 = metrics["classification_metrics"]["f1"]["value"]

    print(f"Test rows: {len(test_df)}")
    print(f"F1: {f1:.6f}")
    print(f"Evaluation artifact: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()