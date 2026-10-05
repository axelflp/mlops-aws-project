import argparse

import joblib
import pandas as pd

from mlops_aws.evaluation import evaluate_model, save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_file)
    model = joblib.load(args.model_file)

    metrics = evaluate_model(model, test_df)

    save_metrics(metrics, args.output_file)


if __name__ == "__main__":
    main()