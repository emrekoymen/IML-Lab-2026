from time import perf_counter
import inspect
import tracemalloc
from sklearn.metrics import accuracy_score, f1_score


def benchmark_classifier(model, X_train, y_train, X_test, y_test):
    tracemalloc.start()

    try:
        # Fitting time
        start = perf_counter()
        model.fit(X_train, y_train)
        training_time = perf_counter() - start

        # Prediction time
        start = perf_counter()
        sig = inspect.signature(model.predict)
        if "y_test" in sig.parameters:
            predictions = model.predict(X_test, y_test=y_test)
        else:
            predictions = model.predict(X_test)
        prediction_time = perf_counter() - start

        _, peak_memory = tracemalloc.get_traced_memory()

        return {
            "predictions": predictions,
            "training_time": training_time,
            "prediction_time": prediction_time,
            "peak_memory_mb": peak_memory / (1024 ** 2),
            "accuracy": accuracy_score(y_test, predictions),
            "f1_macro": f1_score(y_test, predictions, average="macro"),
        }
    finally:
        tracemalloc.stop()
