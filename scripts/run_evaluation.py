import argparse
from datetime import UTC, datetime
from pathlib import Path

import boto3
import yaml
from sagemaker.core.helper.session_helper import Session

from mlops_aws.sagemaker.evaluation_processing import (
    create_evaluation_processor,
    run_evaluation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--role-name", required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
    )

    args = parser.parse_args()

    with (PROJECT_ROOT/args.config).open() as file:
        config = yaml.safe_load(file)

    boto_session = boto3.Session(
        region_name=args.region,
    )

    iam = boto_session.client("iam")

    role_arn = iam.get_role(
        RoleName=args.role_name,
    )["Role"]["Arn"]

    sagemaker_session = Session(
        boto_session=boto_session,
        default_bucket=args.bucket,
    )

    timestamp = datetime.now(UTC).strftime(
        "%Y%m%d-%H%M%S"
    )

    job_name = f"mlops-aws-evaluate-{timestamp}"

    output_s3_uri = (
        f"s3://{args.bucket}/"
        f"{config['output_prefix']}/{job_name}"
    )

    processor = create_evaluation_processor(
        role_arn=role_arn,
        region=args.region,
        instance_type=config["instance_type"],
        instance_count=config["instance_count"],
        volume_size_gb=config["volume_size_gb"],
        max_runtime_seconds=config["max_runtime_seconds"],
        framework_version=config["framework_version"],
        sagemaker_session=sagemaker_session,
    )

    run_evaluation(
        processor=processor,
        source_dir=str(PROJECT_ROOT/"src"),
        model_s3_uri=args.model_uri,
        test_s3_uri=(
            f"s3://{args.bucket}/data/processed/test/test.csv"
        ),
        output_s3_uri=output_s3_uri,
        job_name=job_name,
    )

    print(f"Evaluation job: {job_name}")
    print(
        "Evaluation artifact: "
        f"{output_s3_uri}/evaluation.json"
    )


if __name__ == "__main__":
    main()