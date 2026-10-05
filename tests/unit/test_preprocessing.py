from mlops_aws.preprocessing import TARGET_COLUMN, create_dataset


def test_create_dataset():
    train_df, test_df = create_dataset()


    assert len(train_df)>0
    assert len(test_df)>0
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in test_df.columns