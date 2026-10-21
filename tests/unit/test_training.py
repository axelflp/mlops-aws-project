import joblib

from mlops_aws.preprocessing import create_dataset
from mlops_aws.training import save_model, train_model


def test_train_model():
    train_df, test_df = create_dataset()

    model = train_model(train_df)

    X_test = test_df.drop(columns=["target"])

    predictions = model.predict(X_test)

    assert len(predictions) == len(test_df)


def test_save_model(tmp_path):
    train_df, _ = create_dataset()

    model = train_model(train_df)

    save_model(
        model=model,
        model_dir=str(tmp_path),
    )

    model_file = tmp_path / "model.joblib"

    assert model_file.exists()

    loaded_model = joblib.load(model_file)

    assert loaded_model is not None