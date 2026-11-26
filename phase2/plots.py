"""Charts for the model comparison. Each file is also embedded in the notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from phase2.models import DISPLAY_NAMES


def _label(model_name: str) -> str:
    return DISPLAY_NAMES.get(model_name, model_name)


def plot_cv_comparison(cv_table, feature_set: str, path: Path) -> None:
    part = cv_table[cv_table["feature_set"] == feature_set].copy()
    part["label"] = part["model"].map(_label)
    metrics = ["precision_mean", "recall_mean", "f1_mean"]
    metric_labels = ["Precision", "Recall", "F1"]
    x_pos = np.arange(len(part))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric, label in zip((-width, 0, width), metrics, metric_labels):
        ax.bar(x_pos + offset, part[metric], width=width, label=label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(part["label"], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("5-fold average at cutoff 0.5")
    ax.set_title("Training folds: precision, recall, and F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_set_gap(cv_table, path: Path) -> None:
    part = cv_table[cv_table["model"] != "majority_class"].copy()
    models = list(part[part["feature_set"] == "user_inputs"]["model"])
    labels = [_label(model) for model in models]
    user_f1 = part[part["feature_set"] == "user_inputs"].set_index("model").loc[models, "f1_mean"]
    full_f1 = part[part["feature_set"] == "all_features"].set_index("model").loc[models, "f1_mean"]
    x_pos = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width / 2, user_f1, width=width, label="Without visit rates")
    ax.bar(x_pos + width / 2, full_f1, width=width, label="With PageValues, BounceRates, ExitRates")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("5-fold average F1")
    ax.set_title("What the three visit-rate columns add")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(reports: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(reports), figsize=(5.2 * len(reports), 4.4))
    if len(reports) == 1:
        axes = [axes]
    for ax, report in zip(axes, reports):
        matrix = np.array(
            [[report["tn"], report["fp"]], [report["fn"], report["tp"]]],
            dtype=float,
        )
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["Predicted no", "Predicted yes"])
        ax.set_yticks([0, 1], ["Did not buy", "Bought"])
        for row in range(2):
            for col in range(2):
                ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")
        ax.set_title(report["title"])
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_threshold_sweep(sweep, chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision")
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall")
    ax.plot(sweep["threshold"], sweep["f1"], label="F1")
    ax.axvline(chosen, color="black", linestyle="--", label=f"Chosen cutoff {chosen:.2f}")
    ax.set_xlabel("Probability cutoff")
    ax.set_ylabel("Out-of-fold score")
    ax.set_ylim(0, 1)
    ax.set_title("Cutoff chosen on training folds, not on the test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(curve_rows: list[dict], path: Path, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in curve_rows:
        y_true = np.asarray(row["y_true"])
        proba = np.asarray(row["proba"])
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y_true, proba)
            score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{row['label']} (AUC {score:.3f})")
        else:
            precision, recall, _ = precision_recall_curve(y_true, proba)
            ax.plot(recall, precision, label=row["label"])
    if kind == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No ranking skill")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Test ROC curves")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Test precision-recall curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(ranked: pd.DataFrame, path: Path) -> None:
    part = ranked.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(part["feature"], part["importance_mean"])
    ax.set_xlabel("Drop in test F1 when the column is shuffled")
    ax.set_title("Which inputs the shipped model uses")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def rank_features(pipeline, x_frame, y_true, columns, threshold: float) -> pd.DataFrame:
    """Shuffle each raw input on the test rows and record the F1 drop."""

    def f1_at_cutoff(estimator, features, target):
        proba = estimator.predict_proba(features)[:, 1]
        predicted = (proba >= threshold).astype(int)
        from sklearn.metrics import f1_score

        return f1_score(target, predicted, zero_division=0)

    result = permutation_importance(
        pipeline,
        x_frame[columns],
        y_true,
        n_repeats=5,
        random_state=42,
        scoring=f1_at_cutoff,
        n_jobs=-1,
    )
    ranked = pd.DataFrame(
        {
            "feature": columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return ranked.sort_values("importance_mean", ascending=False).reset_index(drop=True)
