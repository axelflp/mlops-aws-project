from sagemaker.core import image_uris
from sagemaker.core.processing import FrameworkProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)


def create_training_processor(
    *,
    role_arn: str,
    region: str,
    instance_type: str,
    instance_count: int,
    volume_size_gb: int,
    max_runtime_seconds: int,
    framework_version: str,
    sagemaker_session,
):
    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version=framework_version,
        py_version="py3",
        instance_type=instance_type,
    )

    return FrameworkProcessor(
        image_uri=image_uri,
        role=role_arn,
        instance_type=instance_type,
        instance_count=instance_count,
        volume_size_in_gb=volume_size_gb,
        max_runtime_in_seconds=max_runtime_seconds,
        command=["python3"],
        base_job_name="mlops-aws-train",
        sagemaker_session=sagemaker_session,
        tags=[
            {
                "Key": "Project",
                "Value": "mlops-aws-sagemaker",
            },
            {
                "Key": "Environment",
                "Value": "lab",
            },
            {
                "Key": "ManagedBy",
                "Value": "mlops-project",
            },
        ],
    )


def run_training(
    *,
    processor,
    source_dir: str,
    input_s3_uri: str,
    output_s3_uri: str,
    max_iter: int,
    random_state: int,
    job_name: str,
):
    return processor.run(
        code="train_processing_job.py",
        source_dir=source_dir,
        inputs=[
            ProcessingInput(
                input_name="train",
                s3_input=ProcessingS3Input(
                    s3_uri=input_s3_uri,
                    local_path="/opt/ml/processing/input",
                    s3_data_type="S3Prefix",
                ),
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                s3_output=ProcessingS3Output(
                    s3_uri=output_s3_uri,
                    local_path="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob",
                ),
            )
        ],
        arguments=[
            "--max-iter",
            str(max_iter),
            "--random-state",
            str(random_state),
        ],
        job_name=job_name,
        wait=True,
        logs=True,
    )