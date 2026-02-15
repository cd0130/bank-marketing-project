
# app.py — Streamlit app (prediction-only)
# Features:
#   a) Upload TEST CSV (only)
#   b) Model selection dropdown (multiple models, full names)
#   c) Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
#   d) Confusion matrix (compact rendering)
#   e) Classification report (expanded at the bottom, with brief metric explanations)
#
# Assumptions:
#   • Pretrained pipelines (*.joblib) saved under ./model
#   • Target column is 'y'
#   • Uploaded TEST CSV matches the training schema

import io
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# -----------------------------
# Paths (absolute, based on this script)
# -----------------------------
BASE_DIR  = Path(__file__).parent.resolve()
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(page_title="ML Classifier - Chinmay Das (2025AA05677)", page_icon="📈", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def _coerce_target(series: pd.Series) -> pd.Series:
    """Map common string labels to {0,1} and return int series if possible."""
    s = series.copy()
    if s.dtype == "O":
        s = s.astype(str).str.strip().str.lower().map({
            "yes": 1, "no": 0,
            "true": 1, "false": 0,
            "y": 1, "n": 0
        }).fillna(series)
    try:
        s = pd.to_numeric(s)
    except Exception:
        pass
    return s

@st.cache_resource(show_spinner=False)
def load_models(model_dir: Path) -> Dict[str, object]:
    """Load all *.joblib pipelines from model_dir, returns {model_name: pipeline}."""
    models: Dict[str, object] = {}
    if not model_dir.is_dir():
        return models
    for path in sorted(model_dir.glob("*.joblib")):
        name = path.stem
        friendly = name[len("model_"):] if name.startswith("model_") else name
        try:
            models[friendly] = joblib.load(str(path))
        except Exception as e:
            st.warning(f"Failed to load {path.name}: {e}")
    return models

def guess_sep_from_buffer(file_like: io.BytesIO, default: str = ",") -> str:
    """Peek into the start of the uploaded file and guess ',' vs ';' by counting."""
    pos = file_like.tell()
    file_like.seek(0)
    head = file_like.read(2048).decode("utf-8", errors="ignore")
    file_like.seek(pos)  # restore
    commas = head.count(",")
    semis  = head.count(";")
    if semis > commas:
        return ";"
    if commas > semis:
        return ","
    return default

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> pd.DataFrame:
    """Return a one-row DataFrame with required metrics."""
    from sklearn.metrics import matthews_corrcoef
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = np.nan
    else:
        auc = np.nan
    return pd.DataFrame({"Accuracy":[acc], "AUC":[auc], "Precision":[prec], "Recall":[rec], "F1":[f1], "MCC":[mcc]})

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix", width_px: int = 320):
    """Render a compact confusion matrix image at a fixed pixel width."""
    sns.set_context("paper", font_scale=0.75)

    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=140)
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 8}
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    plt.close(fig)
    st.image(buf, caption=None, width=width_px)

# -----------------------------
# Sidebar: Overview
# -----------------------------
with st.sidebar:
    st.header("About this app")
    st.markdown(
        """
**What this app does**
- Loads **pre‑trained classifiers** (Logistic Regression, Decision Tree, K‑Nearest Neighbors, Naive Bayes, Random Forest, XGBoost if present).
- Lets you **upload a Test CSV**, evaluates it, and shows **metrics** and **confusion matrix**.
- The **classification report** is shown at the bottom (expanded).

**Dataset used for training (offline)**
- **UCI Bank Marketing** (semicolon `;` separated).  
- Target column: **`y`** (mapped to 0/1).  
- Note: **`duration`** is dropped during training to avoid leakage.

**How to use**
1. **Download** the small sample test file.
2. **Upload** your Test CSV (same schema as training features).
3. **Pick a model** and view **metrics** + **confusion matrix** + **classification report**.

**Notes**
- CSV separator is **auto‑detected** (`,` or `;`) when you upload.
- Models and small test data are loaded once using Streamlit cache.
        """
    )

# -----------------------------
# Main UI
# -----------------------------
st.title("📈 ML Classifier - Chinmay Das (2025AA05677)")
st.caption("Upload **Test CSV** → Pick a **Model** → See **Metrics**, **Confusion Matrix**, and **Classification report**")

# --- Quick Download for Sample test data (absolute path) ---
st.subheader("📥 Download: Sample test data")
test_csv_path = DATA_DIR / "test_sample.csv"

if test_csv_path.exists():
    @st.cache_data(show_spinner=False)
    def _load_test_bytes(path: Path) -> bytes:
        return path.read_bytes()

    st.download_button(
        label="⬇️ Download sample test CSV",
        data=_load_test_bytes(test_csv_path),
        file_name="test_sample.csv",
        mime="text/csv",
        width="content"
    )
else:
    st.info("No sample test CSV found at `data/test_sample.csv`.")

st.divider()

# Load pretrained models from MODEL_DIR
models = load_models(MODEL_DIR)
if not models:
    st.error("No models found in ./model. Please add joblib pipelines (e.g., model_logreg.joblib) and rerun.")
    st.stop()

# --- Model selection dropdown: show full names ---
pretty_names = {
    "logreg": "Logistic Regression",
    "tree":   "Decision Tree",
    "knn":    "K-Nearest Neighbors",
    "nb":     "Naive Bayes",
    "rf":     "Random Forest (Ensemble)",
    "xgb":    "XGBoost (Ensemble)",
}
internal_to_pretty = {k: pretty_names.get(k, k) for k in models.keys()}
pretty_to_internal = {v: k for k, v in internal_to_pretty.items()}

model_pretty = st.selectbox(
    "Select a model",
    options=list(pretty_to_internal.keys()),
    index=0
)

model_key = pretty_to_internal[model_pretty]
model = models[model_key]

# --- Upload TEST CSV (only) ---
st.caption("Don’t have a TEST CSV handy? Use the **Download sample test CSV** button above, then upload it here.")
uploaded = st.file_uploader("Upload TEST CSV (must contain features; target column optional)", type=["csv"])

target_col = "y"  # fixed

if uploaded is not None:
    try:
        # Auto-detect separator (no UI selector shown)
        autodetected_sep = guess_sep_from_buffer(uploaded, default=",")
        uploaded.seek(0)

        # If you prefer no auto-detect, you can hard-code semicolon here:
        # df_test = pd.read_csv(uploaded, sep=";")
        df_test = pd.read_csv(uploaded, sep=autodetected_sep)

        st.success(
            f"Loaded TEST CSV: {df_test.shape[0]} rows × {df_test.shape[1]} cols  •  Detected separator: **'{autodetected_sep}'**"
        )
        st.dataframe(df_test.head(), width="stretch")

        # Split features/target
        has_target = target_col in df_test.columns
        if has_target:
            y_true = _coerce_target(df_test[target_col]).astype(int)
            X_test = df_test.drop(columns=[target_col])
        else:
            X_test = df_test.copy()

        # Predict
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                y_proba = y_proba[:, 1]
        y_pred = model.predict(X_test)

        # (c) Metrics
        if has_target:
            metrics_df = compute_metrics(y_true.values, y_pred, y_proba)
            st.subheader("Evaluation Metrics (TEST CSV)")
            st.dataframe(metrics_df.style.format("{:.4f}"), width="content")

            # (d) Confusion matrix — compact
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(y_true.values, y_pred, title=f"{model_pretty} — Confusion Matrix")

            # (e) Classification report — at bottom, expanded
            with st.expander("Classification report", expanded=True):
                rep = classification_report(y_true, y_pred, digits=4)
                st.code(rep, language="text")
                st.markdown("""
**What these metrics mean:**
- **Precision**: Of the samples predicted as a given class, how many were actually correct (TP / (TP + FP)).
- **Recall**: Of the samples that truly belong to a class, how many did the model correctly find (TP / (TP + FN)).
- **F1-score**: Harmonic mean of precision and recall; balances both (F1 = 2 · (Precision · Recall) / (Precision + Recall)).
- **Support**: Number of true instances for each class in the test set.
- **Accuracy**: Overall proportion of correct predictions across all classes.
- **Macro avg**: Unweighted average of metrics across classes (treats each class equally).
- **Weighted avg**: Average of metrics weighted by each class’s support (accounts for class imbalance).
                """)

        else:
            st.info("No target column found — showing predictions only.")
            out = X_test.copy()
            out["pred"] = y_pred
            if y_proba is not None:
                out["pred_proba_1"] = y_proba
            st.dataframe(out.head(), width="stretch")
            st.download_button(
                "Download Predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                width="content"
            )

    except Exception as e:
        st.error("Failed to process the uploaded TEST CSV.")
        st.exception(e)
else:
    st.info("Upload a TEST CSV to evaluate the selected model.")