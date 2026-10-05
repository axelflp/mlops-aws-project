import argparse

import pandas as pd

from mlops_aws.training import save_model, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file)

    model = train_model(train_df)

    save_model(model, args.model_dir)


if __name__ == "__main__":
    main()