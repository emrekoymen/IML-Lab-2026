"""Read the ARFF fold files."""

import pandas as pd
from scipy.io import arff


def load_arff(path):
    """Return the features (DataFrame) and the class labels (Series, last column)."""
    data, metadata = arff.loadarff(path)
    table = pd.DataFrame(data)

    for column in table.columns:
        if metadata[column][0] == "nominal":
            # SciPy gives categorical values as bytes, so turn them into text
            table[column] = table[column].str.decode("utf-8")
            table[column] = table[column].replace("?", float("nan"))

    return table.iloc[:, :-1], table.iloc[:, -1]


def load_fold(folder, fold):
    """Load the train and test files of one fold, e.g. hypothyroid.fold.000003.train.arff."""
    name = folder.rstrip("/").split("/")[-1]
    prefix = f"{folder}/{name}.fold.{fold:06d}"
    X_train, y_train = load_arff(prefix + ".train.arff")
    X_test, y_test = load_arff(prefix + ".test.arff")
    return X_train, y_train, X_test, y_test
