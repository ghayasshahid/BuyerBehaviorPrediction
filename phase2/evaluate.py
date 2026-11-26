"""Cross-validation, tuning, threshold choice, and test metrics.

The test rows are not used to pick a model, a parameter, or a threshold.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from phase2.models import build_pipeline
from phase2.preprocessing import RANDOM_STATE


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def safe_roc_auc(y_true, proba) -> float:
    try:
        return float(roc_auc_score(y_true, proba))
    except ValueError:
        return float("nan")


def safe_average_precision(y_true, proba) -> float:
    try:
        return float(average_precision_score(y_true, proba))
    except ValueError:
        return float("nan")


def scores_at_threshold(y_true, proba, threshold: float) -> dict:
    """Precision and recall at one cutoff. Accuracy is recorded, not used to choose."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": safe_roc_auc(y_true, proba),
        "average_precision": safe_average_precision(y_true, proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cross_validate_models(x_train, y_train, model_names, column_sets) -> pd.DataFrame:
    """Average precision, recall, and F1 across 5 stratified folds. Cutoff is 0.5."""
    rows = []
    cv = make_cv()
    for set_name, columns in column_sets.items():
        features = x_train[columns]
        for model_name in model_names:
            print(f"CV {set_name} / {model_name}")
            fold_rows = []
            for train_index, valid_index in cv.split(features, y_train):
                pipeline = build_pipeline(model_name, columns)
                pipeline.fit(features.iloc[train_index], y_train.iloc[train_index])
                proba = pipeline.predict_proba(features.iloc[valid_index])[:, 1]
                fold_rows.append(scores_at_threshold(y_train.iloc[valid_index], proba, 0.5))
            folds = pd.DataFrame(fold_rows)
            row = {"feature_set": set_name, "model": model_name, "folds": int(len(folds))}
            for metric in ["precision", "recall", "f1", "accuracy", "roc_auc", "average_precision"]:
                row[f"{metric}_mean"] = float(folds[metric].mean())
                row[f"{metric}_std"] = float(folds[metric].std(ddof=0))
            rows.append(row)
    return pd.DataFrame(rows)


def tune_model(model_name, columns, param_grid, x_train, y_train) -> dict:
    """Grid search on training rows only. The score being maximized is F1."""
    print(f"Tune {model_name} on {len(columns)} columns")
    pipeline = build_pipeline(model_name, columns)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=make_cv(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train[columns], y_train)
    best_params = {key: _plain(value) for key, value in search.best_params_.items()}
    return {
        "model": model_name,
        "best_params": best_params,
        "cv_f1_mean": float(search.best_score_),
        "estimator": search.best_estimator_,
    }


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def out_of_fold_proba(model_name, columns, params, x_train, y_train) -> np.ndarray:
    """Probabilities from rows the fold did not train on."""
    pipeline = build_pipeline(model_name, columns)
    pipeline.set_params(**params)
    proba = np.zeros(len(y_train), dtype=float)
    features = x_train[columns]
    for train_index, valid_index in make_cv().split(features, y_train):
        fold_model = clone(pipeline)
        fold_model.fit(features.iloc[train_index], y_train.iloc[train_index])
        proba[valid_index] = fold_model.predict_proba(features.iloc[valid_index])[:, 1]
    return proba


def threshold_sweep(y_true, proba) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.15, 0.86, 0.01), 2)
    rows = [scores_at_threshold(y_true, proba, float(threshold)) for threshold in thresholds]
    sweep = pd.DataFrame(rows)
    sweep["distance_from_half"] = (sweep["threshold"] - 0.5).abs()
    return sweep


def choose_threshold(sweep: pd.DataFrame) -> float:
    """Highest F1 on out-of-fold training scores. Ties prefer higher recall."""
    ranked = sweep.sort_values(
        ["f1", "recall", "distance_from_half"],
        ascending=[False, False, True],
    )
    return float(ranked.iloc[0]["threshold"])


def error_slices(x_frame, y_true, proba, threshold: float) -> pd.DataFrame:
    """Where recall is weak on the test rows. This does not change the model."""
    frame = x_frame.copy()
    frame["actual"] = np.asarray(y_true).astype(int)
    frame["predicted"] = (np.asarray(proba) >= threshold).astype(int)
    frame["page_value_group"] = np.where(frame["PageValues"] > 0, "PageValues > 0", "PageValues = 0")
    frame["product_pages"] = pd.cut(
        frame["ProductRelated"],
        bins=[-0.1, 0, 5, 20, 10_000],
        labels=["0 pages", "1-5 pages", "6-20 pages", "21+ pages"],
    )
    frame["weekend_label"] = frame["Weekend"].map({0: "weekday", 1: "weekend"})
    specs = [
        ("VisitorType", "VisitorType"),
        ("Weekend", "weekend_label"),
        ("Month", "Month"),
        ("PageValues", "page_value_group"),
        ("ProductRelated", "product_pages"),
    ]
    rows = []
    for slice_name, column in specs:
        for label, part in frame.groupby(column, observed=True):
            buyers = int(part["actual"].sum())
            flagged = int(part["predicted"].sum())
            rows.append(
                {
                    "slice": slice_name,
                    "group": str(label),
                    "rows": int(len(part)),
                    "buyers": buyers,
                    "flagged": flagged,
                    "recall": float(recall_score(part["actual"], part["predicted"], zero_division=0)) if buyers else float("nan"),
                    "precision": float(precision_score(part["actual"], part["predicted"], zero_division=0)) if flagged else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["slice", "group"]).reset_index(drop=True)
