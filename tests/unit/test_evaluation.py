from mlops_aws.evaluation import evaluate_model
from mlops_aws.preprocessing import create_dataset
from mlops_aws.training import train_model


def test_evaluation_contains_f1():
    train_df, test_df = create_dataset()

    model = train_model(train_df)

    metrics = evaluate_model(model, test_df)

    f1 = metrics["classification_metrics"]["f1"]["value"]

    assert 0 <= f1 <= 1