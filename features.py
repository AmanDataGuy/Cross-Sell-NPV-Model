"""Shared feature prep so model.py / segment.py / backtest.py encode the
same way instead of drifting apart. Not used by hypothesis.py, which needs
the raw (un-dummied) columns for chi-square/t-tests."""
import pandas as pd

TARGET = "y"
DROP = ["customer_id", TARGET]


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.get_dummies(df.drop(columns=DROP), drop_first=True)
    return X, df[TARGET]
