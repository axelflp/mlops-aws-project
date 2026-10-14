import argparse
from pathlib import Path

import pandas as pd

from mlops_aws.preprocessing import save_dataset, split_dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-file", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    dataframe = pd.read_csv(args.input_file)

    train_df, test_df = split_dataset(
        dataframe,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    save_dataset(
        train_df,
        Path(args.train_dir)/"train.csv",
    )

    save_dataset(
        test_df,
        Path(args.test_dir)/"test.csv",
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

if __name__ == '__main__':
    main()