import argparse
from io import StringIO

import boto3
from sklearn.datasets import load_breast_cancer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", required=True)

    args = parser.parse_args()

    dataset = load_breast_cancer(as_frame=True)
    dataframe = dataset.frame

    csv_buffer = StringIO()

    dataframe.to_csv(csv_buffer, index=False)

    s3 = boto3.client(
        "s3",
        region_name=args.region,
    )

    s3.put_object(
        Bucket=args.bucket,
        Key="data/raw/breast_cancer.csv",
        Body=csv_buffer.getvalue(),
        ContentType="text/csv"
    )

    print(f"Uploaded to s3://{args.bucket}/data/raw/breast_cancer.csv")


if __name__ == '__main__':
    main()