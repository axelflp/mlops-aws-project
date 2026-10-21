import json

from mlops_aws.evaluation import evaluate_model, save_metrics
from mlops_aws.preprocessing import create_dataset
from mlops_aws.training import train_model


def test_evaluation_contains_f1():
    train_df, test_df = create_dataset()

    model = train_model(train_df)

    metrics = evaluate_model(model, test_df)

    f1 = metrics["classification_metrics"]["f1"]["value"]

    assert 0 <= f1 <= 1

def test_evaluation_output_contract(tmp_path):
    train_df, test_df = create_dataset()

    model = train_model(train_df)

    metrics = evaluate_model(model, test_df)

    output_file = tmp_path/"evaluation.json"

    save_metrics(
        metrics=metrics,
        output_file=str(output_file),
    )

    with output_file.open() as file:
        result = json.load(file)

    assert "classification_metrics" in result
    assert "f1" in result["classification_metrics"]
    assert "value" in result["classification_metrics"]["f1"]