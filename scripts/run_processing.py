import argparse
from pathlib import Path

import boto3
import yaml
from sagemaker.core.helper.session_helper import Session

from mlops_aws.sagemaker.processing import (
    create_preprocessing_processor,
    run_preprocessing,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--role-name", required=True)
    parser.add_argument(
        "--config",
        default="configs/processing.yaml"
    )

    args = parser.parse_args()

    config_path = PROJECT_ROOT/args.config

    with config_path.open() as file:
        config = yaml.safe_load(file)

    boto_session = boto3.Session(
        region_name=args.region,
    )

    iam = boto_session.client("iam")

    role_arn = iam.get_role(
        RoleName=args.role_name
    )["Role"]["Arn"]

    sagemaker_session = Session(
        boto_session=boto_session,
        default_bucket=args.bucket,
    )

    processor = create_preprocessing_processor(
        role_arn=role_arn,
        region=args.region,
        instance_type=config["instance_type"],
        instance_count=config["instance_count"],
        volume_size_gb=config["volume_size_gb"],
        max_runtime_seconds=config["max_runtime_seconds"],
        framework_version=config["framework_version"],
        sagemaker_session=sagemaker_session,
    )

    run_preprocessing(
        processor=processor,
        source_dir=str(PROJECT_ROOT/"src"),
        input_s3_uri=(
            f"s3://{args.bucket}/data/raw/breast_cancer.csv"
        ),
        train_output_s3_uri=(
            f"s3://{args.bucket}/data/processed/train"
        ),
        test_output_s3_uri=(
            f"s3://{args.bucket}/data/processed/test"
        ),
        test_size=config["test_size"],
        random_state=config["random_state"],
    )


if __name__ == '__main__':
    main()