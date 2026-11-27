"""Fit the Phase 2 models and write the tables, charts, and pipelines.

Run from the repo root:

    py -3 -m phase2.train
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from phase2.evaluate import (
    choose_threshold,
    cross_validate_models,
    error_slices,
    out_of_fold_proba,
    scores_at_threshold,
    threshold_sweep,
    tune_model,
)
from phase2.models import DISPLAY_NAMES, MODEL_ORDER, PARAM_GRIDS, TUNED_MODELS
from phase2.plots import (
    plot_confusion,
    plot_curves,
    plot_cv_comparison,
    plot_feature_set_gap,
    plot_importance,
    plot_threshold_sweep,
    rank_features,
)
from phase2.preprocessing import (
    ALL_COLUMNS,
    CALCULATED_COLUMNS,
    CATEGORY_COLUMNS,
    COLUMN_SETS,
    LOG_COLUMNS,
    USER_COLUMNS,
    load_splits,
)


def _round_row(row: dict) -> dict:
    cleaned = {}
    for key, value in row.items():
        if isinstance(value, float):
            cleaned[key] = None if pd.isna(value) else round(value, 4)
        else:
            cleaned[key] = value
    return cleaned


def _examples(x_train, y_train, app, full) -> dict:
    """One real training buyer and non-buyer that both models get right out of fold.

    Of those rows, the one with the median shipped-model score in its class is used.
    A row of column medians is not a real visit, and the shipped model flagged the
    median non-buyer, so the examples are real rows instead.
    """
    app_scores = pd.Series(app["oof"], index=x_train.index)
    full_scores = pd.Series(full["oof"], index=x_train.index)

    def one_example(label):
        right = ((app_scores >= app["threshold"]) == bool(label)) & (
            (full_scores >= full["threshold"]) == bool(label)
        )
        part = app_scores[(y_train == label) & right]
        index = (part - part.median()).abs().idxmin()
        row = {}
        for column in ALL_COLUMNS:
            value = x_train.at[index, column]
            if column in CATEGORY_COLUMNS:
                row[column] = str(value)
            elif column == "Weekend":
                row[column] = int(value)
            else:
                row[column] = round(float(value), 4)
        return row

    return {
        "likely_buyer": one_example(1),
        "likely_non_buyer": one_example(0),
    }


def _slice_sentence(slices: pd.DataFrame) -> str:
    usable = slices[(slices["buyers"] >= 20) & (slices["slice"] != "Weekend")].copy()
    if usable.empty:
        return "The test slices are too small to name a stable failure group."
    worst = usable.sort_values(["recall", "buyers"], ascending=[True, False]).iloc[0]
    best = usable.sort_values(["recall", "buyers"], ascending=[False, False]).iloc[0]
    return (
        f"On the test set the shipped model is weakest for {worst['slice']} = {worst['group']} "
        f"(recall {worst['recall']:.3f}, {int(worst['buyers'])} buyers) and strongest for "
        f"{best['slice']} = {best['group']} (recall {best['recall']:.3f}, {int(best['buyers'])} buyers)."
    )


def _pagevalue_sentence(slices: pd.DataFrame) -> str:
    part = slices[slices["slice"] == "PageValues"]
    if part.empty:
        return ""
    zero = part[part["group"] == "PageValues = 0"].iloc[0]
    nonzero = part[part["group"] == "PageValues > 0"].iloc[0]
    return (
        f"Recall is close in both PageValues groups ({zero['recall']:.3f} when it is 0, "
        f"{int(zero['buyers'])} buyers; {nonzero['recall']:.3f} when it is above 0, "
        f"{int(nonzero['buyers'])} buyers). Precision is not. "
        f"A flag on a PageValues = 0 visit is right only {zero['precision']:.3f} of the time "
        f"({int(zero['flagged'])} flags). A flag on a PageValues > 0 visit is right "
        f"{nonzero['precision']:.3f} of the time. The shipped model never sees PageValues, "
        "and most of its false alarms sit in the zero group."
    )


def _narrative(summary: dict, slices: pd.DataFrame) -> str:
    app = summary["app_model"]
    full = summary["full_model"]
    test = app["test"]
    half = app["test_at_0_5"]
    lines = [
        (
            f"The shipped model is {DISPLAY_NAMES[app['model']]} on the user-input columns. "
            f"It had the highest 5-fold F1 on the training rows ({app['cv_f1_mean']:.3f}). "
            f"The cutoff {app['threshold']:.2f} was chosen from out-of-fold training scores."
        ),
        (
            f"On the held-out test set ({summary['test_rows']} visits, {summary['test_buyers']} buyers) "
            f"precision is {test['precision']:.3f}, recall is {test['recall']:.3f}, and F1 is {test['f1']:.3f}. "
            f"ROC AUC is {test['roc_auc']:.3f}. Accuracy is {test['accuracy']:.3f}. "
            f"A guess of nobody buys scores {summary['majority_accuracy']:.3f} accuracy and finds 0 of the "
            f"{summary['test_buyers']} buyers, so accuracy is the wrong ranking score."
        ),
        (
            f"At the default cutoff 0.5 the same model has precision {half['precision']:.3f}, "
            f"recall {half['recall']:.3f}, and F1 {half['f1']:.3f}. "
            f"The training-fold cutoff moves that tradeoff. Test F1 at the chosen cutoff is {test['f1']:.3f}."
        ),
        (
            f"Training-row F1 at the same cutoff is {app['train_f1']:.3f}, against test F1 {test['f1']:.3f}. "
            + (
                "The gap is large enough to treat the training score as optimistic."
                if app["train_f1"] - test["f1"] > 0.08
                else "The training and test F1 scores are close, so the fit is not memorizing the training rows."
            )
        ),
        (
            f"The best model that is also allowed to see {', '.join(CALCULATED_COLUMNS)} is "
            f"{DISPLAY_NAMES[full['model']]}. Its 5-fold F1 is {full['cv_f1_mean']:.3f} and its test F1 is "
            f"{full['test']['f1']:.3f} (precision {full['test']['precision']:.3f}, recall {full['test']['recall']:.3f}, "
            f"ROC AUC {full['test']['roc_auc']:.3f}). The score on the same rows it trained on is {full['train_f1']:.3f}, "
            "which is higher because that forest is deep. The fold score and the test score agree, so the test number "
            "is the one to trust. Those three columns are calculated from the visit, so the website asks for them only "
            "in an optional analytics section. The higher score is an upper bound, not the form a shopper fills in."
        ),
        _slice_sentence(slices),
        _pagevalue_sentence(slices),
        (
            "What would improve it: a cost for a missed buyer if the shop can say what a false alarm costs, "
            "a second year of visits so February and June are not tiny, and a version that does not need "
            "PageValues if the analytics team can log that score after the visit instead of before the guess."
        ),
    ]
    return " ".join(lines)


def _finish(choice, set_name, fitted, x_train, y_train, x_test, y_test):
    model_name = choice["model"]
    tuned = fitted[(set_name, model_name)]
    columns = COLUMN_SETS[set_name]
    oof = out_of_fold_proba(model_name, columns, tuned["best_params"], x_train, y_train)
    sweep = threshold_sweep(y_train, oof)
    threshold = choose_threshold(sweep)
    estimator = tuned["estimator"]
    test_proba = estimator.predict_proba(x_test[columns])[:, 1]
    train_proba = estimator.predict_proba(x_train[columns])[:, 1]
    test_scores = scores_at_threshold(y_test, test_proba, threshold)
    half_scores = scores_at_threshold(y_test, test_proba, 0.5)
    train_scores = scores_at_threshold(y_train, train_proba, threshold)
    return {
        "model": model_name,
        "feature_set": set_name,
        "columns": columns,
        "best_params": tuned["best_params"],
        "cv_f1_mean": float(choice["cv_f1_mean"]),
        "threshold": threshold,
        "sweep": sweep,
        "estimator": estimator,
        "test_proba": test_proba,
        "test": test_scores,
        "test_at_0_5": half_scores,
        "train_f1": float(train_scores["f1"]),
        "oof": oof,
    }


def run(root: Path | None = None) -> dict:
    root = ROOT if root is None else Path(root)
    results = root / "phase2" / "results"
    model_dir = root / "phase2" / "models"
    results.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_splits(root)
    print(f"Train {len(y_train)} rows, {int(y_train.sum())} buyers")
    print(f"Test {len(y_test)} rows, {int(y_test.sum())} buyers")

    cv_table = cross_validate_models(x_train, y_train, MODEL_ORDER, COLUMN_SETS)
    cv_table.to_csv(results / "cv_metrics.csv", index=False)

    fitted = {}
    tuned_rows = []
    for set_name, columns in COLUMN_SETS.items():
        for model_name in TUNED_MODELS:
            tuned = tune_model(model_name, columns, PARAM_GRIDS[model_name], x_train, y_train)
            fitted[(set_name, model_name)] = tuned
            tuned_rows.append(
                {
                    "feature_set": set_name,
                    "model": model_name,
                    "cv_f1_mean": tuned["cv_f1_mean"],
                    "best_params": json.dumps(tuned["best_params"]),
                }
            )
    tuned_table = pd.DataFrame(tuned_rows).sort_values(
        ["feature_set", "cv_f1_mean"], ascending=[True, False]
    )
    tuned_table.to_csv(results / "tuned_metrics.csv", index=False)

    app_row = tuned_table[tuned_table["feature_set"] == "user_inputs"].iloc[0]
    full_row = tuned_table[tuned_table["feature_set"] == "all_features"].iloc[0]
    app = _finish(app_row, "user_inputs", fitted, x_train, y_train, x_test, y_test)
    full = _finish(full_row, "all_features", fitted, x_train, y_train, x_test, y_test)

    app["sweep"].to_csv(results / "threshold_sweep_app.csv", index=False)
    full["sweep"].to_csv(results / "threshold_sweep_full.csv", index=False)

    slices = error_slices(x_test, y_test, app["test_proba"], app["threshold"])
    slices.to_csv(results / "error_slices.csv", index=False)

    importance = rank_features(
        app["estimator"], x_test, y_test, USER_COLUMNS, app["threshold"]
    )
    importance.to_csv(results / "feature_importance.csv", index=False)

    plot_cv_comparison(cv_table, "user_inputs", results / "04_cv_comparison.png")
    plot_feature_set_gap(cv_table, results / "04_feature_set_gap.png")
    plot_confusion(
        [
            {"title": f"Shipped model, cutoff {app['threshold']:.2f}", **app["test"]},
            {"title": f"With visit rates, cutoff {full['threshold']:.2f}", **full["test"]},
        ],
        results / "04_confusion_matrices.png",
    )
    plot_threshold_sweep(app["sweep"], app["threshold"], results / "04_threshold_sweep.png")
    curve_rows = [
        {"label": "Shipped model", "y_true": y_test, "proba": app["test_proba"]},
        {"label": "With visit rates", "y_true": y_test, "proba": full["test_proba"]},
    ]
    plot_curves(curve_rows, results / "04_roc_curves.png", "roc")
    plot_curves(curve_rows, results / "04_pr_curves.png", "pr")
    plot_importance(importance, results / "04_feature_importance.png")

    examples = _examples(x_train, y_train, app, full)
    shown = pd.DataFrame([examples["likely_buyer"], examples["likely_non_buyer"]], columns=ALL_COLUMNS)
    # The website opens on these rows and either model can score them, so each must
    # land on its own side of the cutoff for both models.
    for result in (app, full):
        buyer_p, non_buyer_p = result["estimator"].predict_proba(shown[result["columns"]])[:, 1]
        assert buyer_p >= result["threshold"] > non_buyer_p, (result["model"], buyer_p, non_buyer_p)
    (results / "example_visits.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")

    def public_model(result):
        return {
            "model": result["model"],
            "display_name": DISPLAY_NAMES[result["model"]],
            "feature_set": result["feature_set"],
            "best_params": result["best_params"],
            "cv_f1_mean": round(result["cv_f1_mean"], 4),
            "threshold": result["threshold"],
            "train_f1": round(result["train_f1"], 4),
            "test": _round_row(result["test"]),
            "test_at_0_5": _round_row(result["test_at_0_5"]),
        }

    summary = {
        "train_rows": int(len(y_train)),
        "train_buyers": int(y_train.sum()),
        "test_rows": int(len(y_test)),
        "test_buyers": int(y_test.sum()),
        "majority_accuracy": round(float((y_test == 0).mean()), 4),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "app_model": public_model(app),
        "full_model": public_model(full),
        "calculated_columns": CALCULATED_COLUMNS,
        "user_columns": USER_COLUMNS,
        "log_columns": LOG_COLUMNS,
    }
    summary["analysis"] = _narrative(summary, slices)
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    joblib.dump(app["estimator"], model_dir / "app_pipeline.joblib")
    joblib.dump(full["estimator"], model_dir / "full_pipeline.joblib")
    print(summary["analysis"])
    print("Saved models to", model_dir)
    return summary


if __name__ == "__main__":
    run()
