import argparse

from mlops_aws.preprocessing import create_dataset, save_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    train_df, test_df = create_dataset()

    save_datasets(
        train_df,
        test_df,
        args.output_dir,
    )

if __name__ == '__main__':
    main()