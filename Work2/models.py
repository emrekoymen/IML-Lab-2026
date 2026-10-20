"""Classification algorithms for Work 2."""

from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from KNNClassifier import kNNAlgorithm


def svmAlgorithm(X_train, y_train, X_test, kernel="rbf", C=1.0, gamma="scale", degree=3):
    """Return predictions, training seconds, and prediction seconds."""
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)

    start = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start

    start = perf_counter()
    predictions = model.predict(X_test)
    prediction_time = perf_counter() - start
    return predictions, training_time, prediction_time


def rfAlgorithm(X_train, y_train, X_test, n_estimators=100, criterion="gini", max_depth=None):
    """Return predictions, training seconds, and prediction seconds."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=42
    )

    start = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start

    start = perf_counter()
    predictions = model.predict(X_test)
    prediction_time = perf_counter() - start
    return predictions, training_time, prediction_time

