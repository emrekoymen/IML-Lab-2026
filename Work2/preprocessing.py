"""Preprocessing: one generic function and one small function per dataset."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def preprocess(X_train, X_test):
    """Fill missing values, scale numbers to [0, 1] and one-hot encode categories."""

    X_train = X_train.copy()
    X_test = X_test.copy()
    numeric = X_train.select_dtypes(include="number").columns
    categorical = X_train.select_dtypes(exclude="number").columns

    # Numbers: median for missing values, then min-max scaling
    if len(numeric) > 0:
        steps = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler(clip=True))
        X_train[numeric] = steps.fit_transform(X_train[numeric])
        X_test[numeric] = steps.transform(X_test[numeric])

    # Categories: most frequent value for missing values
    if len(categorical) > 0:
        imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical] = imputer.fit_transform(X_train[categorical])
        X_test[categorical] = imputer.transform(X_test[categorical])

    # One-hot encoding, test gets the same columns as train
    X_train = pd.get_dummies(X_train, dtype=float)
    X_test = pd.get_dummies(X_test, dtype=float)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    return X_train, X_test


def preprocess_hypothyroid(X_train, X_test):
    # TBG is always missing and TBG_measured is always "f", so they carry no information
    X_train = X_train.drop(columns=["TBG", "TBG_measured"])
    X_test = X_test.drop(columns=["TBG", "TBG_measured"])

    # An age of 455 is a typo, so we treat it as missing
    X_train.loc[X_train["age"] == 455, "age"] = float("nan")
    X_test.loc[X_test["age"] == 455, "age"] = float("nan")

    return preprocess(X_train, X_test)
