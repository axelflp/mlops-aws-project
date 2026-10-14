from sklearn.datasets import load_breast_cancer

from mlops_aws.preprocessing import (
    TARGET_COLUMN,
    create_dataset,
    split_dataset,
)


def test_create_dataset():
    train_df, test_df = create_dataset()


    assert len(train_df)>0
    assert len(test_df)>0
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in test_df.columns


def test_split_dataset():
    dataset = load_breast_cancer(as_frame=True)

    train_df, test_df = split_dataset(dataset.frame)

    assert len(train_df) == 455
    assert len(test_df) == 114

    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in test_df.columns