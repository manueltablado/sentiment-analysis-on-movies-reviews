import os
from typing import Tuple

import gdown
import pandas as pd

from src import config


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Download application_test_aai.csv
    if not os.path.exists(config.DATASET_TEST):
        gdown.download(config.DATASET_TEST_URL, config.DATASET_TEST, quiet=False)

    # Download application_train_aai.csv
    if not os.path.exists(config.DATASET_TRAIN):
        gdown.download(config.DATASET_TRAIN_URL, config.DATASET_TRAIN, quiet=False)

    train = pd.read_csv(config.DATASET_TRAIN)
    test = pd.read_csv(config.DATASET_TEST)

    return train, test


def split_data(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    print(train.columns)
    print(test.columns)

    X_train = train["review"]
    y_train = train["positive"]
    X_test = test["review"]
    y_test = test["positive"]

    return X_train, y_train, X_test, y_test
