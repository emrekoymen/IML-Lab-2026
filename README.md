# Work 2: Classification

Compares SVM and Random Forest (custom kNN coming next) on the Hypothyroid and Soybean datasets, using the supplied 10 folds.

- `parser.py`: reads the ARFF files.
- `preprocessing.py`: fills missing values, scales numbers, one-hot encodes categories.
- `models.py`: the SVM and Random Forest functions.
- `experiments.py`: runs all models on all folds and saves accuracy and plots in `results/`.
- `inspect_data.ipynb`: quick look at the data.

Put the `hypothyroid/` and `soybean/` ARFF folders inside `Work2/`, then run from there:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
cd Work2
python experiments.py
```
