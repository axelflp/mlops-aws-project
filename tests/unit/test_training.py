from mlops_aws.preprocessing import TARGET_COLUMN, create_dataset
from mlops_aws.training import train_model


def test_train_model():
    train_df, test_df = create_dataset()

    model = train_model(train_df)

    X_test = test_df.drop(columns=[TARGET_COLUMN])

    predictions = model.predict(X_test)

    assert len(predictions) == len(test_df)