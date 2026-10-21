from sagemaker.core import image_uris
from sagemaker.core.processing import FrameworkProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)


def create_evaluation_processor(
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
        base_job_name="mlops-aws-evaluate",
        sagemaker_session=sagemaker_session,
        tags={
            "Project":"mlops-aws-sagemaker",
            "Enviroment": "lab",
            "ManagedBy": "mlops-project",
        }
    )


def run_evaluation(
        *,
        processor,
        source_dir,
        model_s3_uri: str,
        test_s3_uri: str,
        output_s3_uri: str,
        job_name: str,
):
    return processor.run(
        code="evaluate_processing_job.py",
        source_dir=source_dir,
        inputs=[
            ProcessingInput(
                input_name="model",
                s3_input=ProcessingS3Input(
                    s3_uri=model_s3_uri,
                    local_path="/opt/ml/processing/model",
                    s3_data_type="S3Prefix",
                ),
            ),
            ProcessingInput(
                input_name="test",
                s3_input=ProcessingS3Input(
                    s3_uri=test_s3_uri,
                    local_path="/opt/ml/processing/test",
                    s3_data_type="S3Prefix",
                ),
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                s3_output=ProcessingS3Output(
                    s3_uri=output_s3_uri,
                    local_path="/opt/ml/processing/evaluation",
                    s3_upload_mode="EndOfJob",
                ),
            )
        ],
        job_name=job_name,
        wait=True,
        logs=True,
    )