import os
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
    # 1. If absolute or relative to CWD and exists, use it directly
    if os.path.exists(folder):
        resolved_folder = folder
    else:
        # 2. Check relative to parser.py's own directory (Work2/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_to_script = os.path.join(current_dir, folder)
        if os.path.exists(relative_to_script):
            resolved_folder = relative_to_script
        elif os.path.exists(os.path.join(current_dir, "Work2", folder)):
            resolved_folder = os.path.join(current_dir, "Work2", folder)
        else:
            resolved_folder = folder

    name = os.path.basename(os.path.normpath(resolved_folder))
    prefix = os.path.join(resolved_folder, f"{name}.fold.{fold:06d}")
    X_train, y_train = load_arff(prefix + ".train.arff")
    X_test, y_test = load_arff(prefix + ".test.arff")
    return X_train, y_train, X_test, y_test
