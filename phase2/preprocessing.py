"""Build the same preparation steps as Phase 1, unfitted.

The log, the scaler, and the category encoder sit inside the model pipeline.
Cross-validation then fits them on each fold's training rows only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

ALL_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

# These three are calculated from the visit. A shopper does not type them.
CALCULATED_COLUMNS = ["BounceRates", "ExitRates", "PageValues"]

LOG_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORY_COLUMNS = [
    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

USER_COLUMNS = [column for column in ALL_COLUMNS if column not in CALCULATED_COLUMNS]

COLUMN_SETS = {
    "user_inputs": USER_COLUMNS,
    "all_features": ALL_COLUMNS,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_category_text(series: pd.Series) -> pd.Series:
    """Store category codes as text, so 2 and "2" are the same label."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.round().astype(int).astype(str)
    return series.astype(str)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in CATEGORY_COLUMNS:
        prepared[column] = _as_category_text(prepared[column])
    for column in LOG_COLUMNS:
        prepared[column] = prepared[column].astype(float)
    prepared["Weekend"] = prepared["Weekend"].astype(int)
    return prepared


def load_splits(root: Path | None = None):
    """Load the Phase 1 split. The raw file is not used here."""
    root = repo_root() if root is None else Path(root)
    train = pd.read_csv(root / "data" / "processed" / "train_clean.csv")
    test = pd.read_csv(root / "data" / "processed" / "test_clean.csv")
    train = _prepare_frame(train)
    test = _prepare_frame(test)

    x_train = train[ALL_COLUMNS].reset_index(drop=True)
    y_train = train["Revenue"].astype(int).reset_index(drop=True)
    x_test = test[ALL_COLUMNS].reset_index(drop=True)
    y_test = test["Revenue"].astype(int).reset_index(drop=True)
    return x_train, y_train, x_test, y_test


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Log and scale numeric columns, and one-hot encode categories.

    `columns` is the set the model is allowed to see. Dropped columns are
    left out, so a model can be trained with or without the visit rates.
    """
    log_columns = [column for column in LOG_COLUMNS if column in columns]
    category_columns = [column for column in CATEGORY_COLUMNS if column in columns]
    transformers = []

    if log_columns:
        log_pipe = Pipeline(
            steps=[
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("log_scale", log_pipe, log_columns))

    if "Weekend" in columns:
        transformers.append(("weekend", "passthrough", ["Weekend"]))

    if category_columns:
        encoder = OneHotEncoder(
            min_frequency=20,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )
        transformers.append(("category", encoder, category_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")
