# Titanic Survival Classification

Predicting whether a passenger survived the Titanic sinking, using their demographic and travel details (class, sex, age, family size, fare, port of embarkation). Built as a comparison between **NumPy-only, from-scratch** implementations and their **scikit-learn** counterparts.

Dataset: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data) (`Titanic-Dataset.csv`, 891 passengers).

## Notebooks

Run in order — each notebook depends on files produced by the previous one.

| Notebook | What it does |
|---|---|
| [`A_minimal_prep_and_split.ipynb`](notebooks/A_minimal_prep_and_split.ipynb) | Drops identifiers/free-text columns, imputes missing `Age`/`Embarked`, encodes categoricals, and produces a stratified 80/20 train/test split. Saves `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`. |
| [`B_from_scratch_models.ipynb`](notebooks/B_from_scratch_models.ipynb) | NumPy-only logistic regression (none/L1/L2 penalties, functional gradient-descent implementation), a small Gini decision tree, bagging, and AdaBoost-style boosting with decision stumps. Saves `section_b_results.csv`. |
| [`C_library_models.ipynb`](notebooks/C_library_models.ipynb) | scikit-learn equivalents — `LogisticRegression` (none/L1/L2), `RandomForestClassifier`, and three boosting classifiers (`AdaBoost`, `GradientBoosting`, `HistGradientBoosting`) — plus a comparison table against Section B and a short analysis/write-up. |

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook
```

Then open and run `A_minimal_prep_and_split.ipynb`, followed by `B_from_scratch_models.ipynb` and `C_library_models.ipynb`.

## Approach

- **Preprocessing:** `PassengerId`, `Name`, `Ticket`, `Cabin` dropped; `Age` median-imputed; `Embarked` mode-imputed; `Sex` binary-mapped; `Embarked` one-hot encoded (`drop_first=True`). Features standardised before modelling.
- **Models:** each model, from-scratch or library, exposes a consistent `fit(X, y)` / `predict_proba(X)` / `predict(X, threshold=0.5)`-style interface for a fair, direct comparison.
- **Evaluation:** accuracy, precision, recall, F1, and ROC-AUC, plus confusion matrices and ROC curves, computed identically across both sections.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| B: Logistic (none) | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| B: Logistic (L1) | 0.804 | 0.793 | 0.667 | 0.724 | 0.843 |
| B: Logistic (L2) | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| B: Bagging | 0.810 | 0.843 | 0.623 | 0.717 | 0.841 |
| B: AdaBoost | 0.804 | 0.774 | 0.696 | 0.733 | 0.829 |
| C: Logistic (none) | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| C: Logistic (L1) | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| C: Logistic (L2) | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| C: Random Forest (Bagging) | 0.804 | 0.870 | 0.580 | 0.696 | 0.847 |
| C: AdaBoost | 0.782 | 0.750 | 0.652 | 0.698 | 0.825 |
| C: Gradient Boosting | 0.799 | 0.789 | 0.652 | 0.714 | 0.818 |
| C: HistGradientBoosting | 0.788 | 0.772 | 0.638 | 0.698 | 0.820 |

All models land around **0.78–0.81 accuracy** and **0.82–0.85 ROC-AUC**, consistent with the known ceiling on this dataset given the available features. From-scratch and library implementations of the same algorithm produce closely matching metrics, which validates the from-scratch code; small gaps come from solver differences (e.g. scikit-learn's `liblinear`/coordinate-descent solver vs. batch gradient descent) and extra randomisation in `RandomForestClassifier` (per-split feature subsetting) that the from-scratch bagging doesn't do.

## Possible next steps

1. Engineer a `Title` feature from `Name` (Mr/Mrs/Miss/Master/...) and a `FamilySize` feature from `SibSp` + `Parch` before dropping the originals.
2. Tune hyperparameters (regularisation strength, tree depth, number of estimators) with cross-validation instead of a single train/test split.
3. Check calibration of predicted probabilities (e.g. a reliability diagram), useful if probabilities are consumed downstream rather than just the class label.
