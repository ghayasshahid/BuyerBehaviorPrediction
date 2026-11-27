"""Score one shopper visit in the browser.

From the repo root:

    py -3 -m streamlit run website/app.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2.preprocessing import ALL_COLUMNS, CALCULATED_COLUMNS

SCHEMA_PATH = ROOT / "data" / "processed" / "input_schema.json"
SUMMARY_PATH = ROOT / "phase2" / "results" / "summary.json"
EXAMPLES_PATH = ROOT / "phase2" / "results" / "example_visits.json"
APP_MODEL_PATH = ROOT / "phase2" / "models" / "app_pipeline.joblib"
FULL_MODEL_PATH = ROOT / "phase2" / "models" / "full_pipeline.joblib"

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
SUMMARY = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
EXAMPLES = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
APP_MODEL = joblib.load(APP_MODEL_PATH)
FULL_MODEL = joblib.load(FULL_MODEL_PATH)


def _number(column: str, fallback: float) -> float:
    value = st.session_state.get(column, fallback)
    return float(value)


def _text(column: str, fallback: str) -> str:
    return str(st.session_state.get(column, fallback))


def apply_example(name: str) -> None:
    """Copy a saved example into the form widgets."""
    example = EXAMPLES[name]
    for column in ALL_COLUMNS:
        st.session_state[column] = example[column]
    st.session_state["use_analytics"] = False


def blank_state() -> None:
    if "Month" in st.session_state:
        return
    apply_example("likely_non_buyer")


def build_row() -> pd.DataFrame:
    row = {}
    for column, info in SCHEMA.items():
        if info["type"] == "category":
            row[column] = _text(column, info["allowed_values"][0])
        elif column == "Weekend":
            row[column] = int(_number(column, 0))
        else:
            row[column] = float(_number(column, info.get("median", 0.0)))
    return pd.DataFrame([row], columns=ALL_COLUMNS)


def score_row(use_analytics: bool) -> dict:
    frame = build_row()
    if use_analytics:
        model = FULL_MODEL
        card = SUMMARY["full_model"]
    else:
        model = APP_MODEL
        card = SUMMARY["app_model"]
    probability = float(model.predict_proba(frame)[:, 1][0])
    return {
        "probability": probability,
        "flagged": probability >= card["threshold"],
        "card": card,
        "frame": frame,
    }


st.set_page_config(page_title="Buying behavior", page_icon="🛒", layout="centered")
blank_state()

st.title("Will this visit end in a purchase?")
st.write(
    "Enter what you know about one visit. The score is the model's estimate that the visitor buys. "
    "About 15.6% of the visits in the training data bought, so a high score is uncommon."
)

left, right = st.columns(2)
if left.button("Example: typical non-buyer", use_container_width=True):
    apply_example("likely_non_buyer")
    st.rerun()
if right.button("Example: typical buyer", use_container_width=True):
    apply_example("likely_buyer")
    st.rerun()

st.subheader("Visit")
form_left, form_right = st.columns(2)

with form_left:
    st.selectbox("Month", SCHEMA["Month"]["allowed_values"], key="Month")
    st.selectbox("Visitor type", SCHEMA["VisitorType"]["allowed_values"], key="VisitorType")
    st.selectbox("Weekend", [0, 1], format_func=lambda value: "Weekend" if value == 1 else "Weekday", key="Weekend")
    st.selectbox("Special day score", SCHEMA["SpecialDay"]["allowed_values"], key="SpecialDay")
    st.number_input("Administrative pages", min_value=0.0, max_value=float(SCHEMA["Administrative"]["max"]), step=1.0, key="Administrative")
    st.number_input("Administrative time (seconds)", min_value=0.0, max_value=float(SCHEMA["Administrative_Duration"]["max"]), step=1.0, key="Administrative_Duration")
    st.number_input("Informational pages", min_value=0.0, max_value=float(SCHEMA["Informational"]["max"]), step=1.0, key="Informational")
    st.number_input("Informational time (seconds)", min_value=0.0, max_value=float(SCHEMA["Informational_Duration"]["max"]), step=1.0, key="Informational_Duration")

with form_right:
    st.number_input("Product pages", min_value=0.0, max_value=float(SCHEMA["ProductRelated"]["max"]), step=1.0, key="ProductRelated")
    st.number_input("Product time (seconds)", min_value=0.0, max_value=float(SCHEMA["ProductRelated_Duration"]["max"]), step=1.0, key="ProductRelated_Duration")
    st.selectbox("Operating system code", SCHEMA["OperatingSystems"]["allowed_values"], key="OperatingSystems")
    st.selectbox("Browser code", SCHEMA["Browser"]["allowed_values"], key="Browser")
    st.selectbox("Region code", SCHEMA["Region"]["allowed_values"], key="Region")
    st.selectbox("Traffic type code", SCHEMA["TrafficType"]["allowed_values"], key="TrafficType")

use_analytics = st.checkbox(
    "I have bounce rate, exit rate, and page value from the shop's analytics",
    key="use_analytics",
)
st.caption("These three numbers are calculated from the visit. They are optional. Turning them on uses the second model.")
# Always drawn, greyed out when off: Streamlit drops the values of widgets that are not drawn,
# which made an example's analytics values revert to the previous visit's.
analytics_left, analytics_mid, analytics_right = st.columns(3)
analytics_left.number_input("Bounce rate", min_value=0.0, max_value=float(SCHEMA["BounceRates"]["max"]), step=0.001, format="%.4f", key="BounceRates", disabled=not use_analytics)
analytics_mid.number_input("Exit rate", min_value=0.0, max_value=float(SCHEMA["ExitRates"]["max"]), step=0.001, format="%.4f", key="ExitRates", disabled=not use_analytics)
analytics_right.number_input("Page value", min_value=0.0, max_value=float(SCHEMA["PageValues"]["max"]), step=0.1, key="PageValues", disabled=not use_analytics)

if st.button("Score this visit", type="primary"):
    result = score_row(use_analytics)
    card = result["card"]
    probability = result["probability"]
    st.subheader("Result")
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Chance of purchase", f"{probability:.1%}")
    metric_right.metric("Cutoff used", f"{card['threshold']:.2f}")
    if result["flagged"]:
        st.success("The model would flag this visit as a likely purchase.")
    else:
        st.info("The model would not flag this visit as a purchase.")
    st.write(
        f"This uses {card['display_name']}. On the held-out test set it had "
        f"precision {card['test']['precision']:.3f}, recall {card['test']['recall']:.3f}, "
        f"and F1 {card['test']['f1']:.3f}."
    )

with st.expander("What the test set showed"):
    app_card = SUMMARY["app_model"]
    full_card = SUMMARY["full_model"]
    st.write(SUMMARY["analysis"])
    st.write(
        f"Without the analytics columns, {app_card['display_name']} found "
        f"{app_card['test']['tp']} of {SUMMARY['test_buyers']} test buyers and raised "
        f"{app_card['test']['fp']} false alarms."
    )
    st.write(
        f"With the analytics columns, {full_card['display_name']} found "
        f"{full_card['test']['tp']} of {SUMMARY['test_buyers']} test buyers and raised "
        f"{full_card['test']['fp']} false alarms."
    )
