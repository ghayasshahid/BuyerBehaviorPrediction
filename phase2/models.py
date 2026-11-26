"""Model definitions.

Class weights are set on the training rows so the 15.6% of buyers are not
ignored. The weights are part of the estimator, so each cross-validation
fold computes them from that fold's training rows only.
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from phase2.preprocessing import RANDOM_STATE, make_preprocessor

MODEL_ORDER = [
    "majority_class",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
]

TUNED_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
]

DISPLAY_NAMES = {
    "majority_class": "Majority class",
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
}

# Small grids. Scoring is F1, so the search balances precision and recall.
PARAM_GRIDS = {
    "logistic_regression": {
        "model__C": [0.1, 1.0, 3.0],
    },
    "decision_tree": {
        "model__max_depth": [4, 8, 12],
        "model__min_samples_leaf": [10, 40],
    },
    "random_forest": {
        "model__max_depth": [10, 18],
        "model__min_samples_leaf": [2, 6],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 6],
    },
}


def build_model(name: str):
    """Return a new estimator. A new object is required for every fit."""
    if name == "majority_class":
        # Always predicts the common class. Accuracy looks high. Recall is zero.
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    raise KeyError(name)


def build_pipeline(name: str, columns: list[str]) -> Pipeline:
    """Preparation and model in one pipeline, so folds cannot leak the scaler."""
    return Pipeline(
        steps=[
            ("prep", make_preprocessor(columns)),
            ("model", build_model(name)),
        ]
    )
