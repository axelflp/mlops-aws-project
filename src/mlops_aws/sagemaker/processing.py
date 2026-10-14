from sagemaker.core import image_uris
from sagemaker.core.processing import FrameworkProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)


def create_preprocessing_processor(
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
        base_job_name="mlops-aws-preprocess",
        sagemaker_session=sagemaker_session,
        tags={
            "Project": "mlops-aws-sagemaker",
            "Enviroment": "lab",
            "ManagedBy": "mlops-project",
            },
    )

def run_preprocessing(
        *,
        processor,
        source_dir: str,
        input_s3_uri: str,
        train_output_s3_uri: str,
        test_output_s3_uri: str,
        test_size: float,
        random_state: int,
):
    return processor.run(
        code="preprocess_job.py",
        source_dir=source_dir,
        inputs=[
            ProcessingInput(
                input_name="raw_data",
                s3_input=ProcessingS3Input(
                    s3_uri=input_s3_uri,
                    local_path="/opt/ml/processing/input",
                    s3_data_type="S3Prefix",
                ),
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                s3_output=ProcessingS3Output(
                    s3_uri=train_output_s3_uri,
                    local_path="/opt/ml/processing/output/train",
                    s3_upload_mode="EndOfJob",
                ),
            ),
            ProcessingOutput(
                output_name="test",
                s3_output=ProcessingS3Output(
                    s3_uri=test_output_s3_uri,
                    local_path="/opt/ml/processing/output/test",
                    s3_upload_mode="EndOfJob",
                ),
            ),
        ],
        arguments=[
            "--input-file",
            "/opt/ml/processing/input/breast_cancer.csv",
            "--train-dir",
            "/opt/ml/processing/output/train",
            "--test-dir",
            "/opt/ml/processing/output/test",
            "--test-size",
            str(test_size),
            "--random-state",
            str(random_state),
        ],
        wait=True,
        logs=True,
    )