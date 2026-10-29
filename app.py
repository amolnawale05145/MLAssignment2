from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# Application configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine (SVM)": "svm.joblib",
}

st.set_page_config(
    page_title="AIML Classification Lab",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Machine Learning Classification Lab")
st.subheader("BITS Pilani WILP | M.Tech AIML | Assignment 2")

st.markdown(
    """
    This application compares five trained classification models using
    a labelled test dataset.

    **Instructions**
    1. Upload a test CSV containing all 30 input features and a `target` column.
    2. Select a model to inspect its detailed performance.
    3. Compare all five models using the evaluation metrics.
    4. Review the confusion matrix and classification report.

    Models are loaded from saved files. The uploaded test data is not
    used to retrain the models.
    """
)

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
with st.sidebar:
    st.header("Evaluation Controls")

    selected_model = st.selectbox(
        "Select a model for detailed evaluation",
        list(MODEL_FILES.keys()),
    )

    uploaded_file = st.file_uploader(
        "Upload labelled test data (CSV)",
        type=["csv"],
    )

    st.caption("Expected target column: target")

    sample_path = ROOT / "test_data.csv"

    if sample_path.exists():
        st.download_button(
            "Download sample test CSV",
            data=sample_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# Load and validate test data
# --------------------------------------------------
if uploaded_file is None:
    st.info("Upload test_data.csv using the sidebar to begin evaluation.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the uploaded CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

if TARGET not in data.columns:
    st.error(
        "The uploaded CSV must contain a column named 'target' "
        "in addition to all required input features."
    )
    st.stop()

if data.isnull().values.any():
    st.error(
        "Missing values detected. Please clean the CSV before uploading."
    )
    st.stop()

# Read the expected feature names from the training data.
train_path = ROOT / "data" / "train_data.csv"

if not train_path.exists():
    st.error(
        "Training data not found. Run `python train_models.py` "
        "from the project directory first."
    )
    st.stop()

expected_features = [
    col
    for col in pd.read_csv(train_path, nrows=0).columns
    if col != TARGET
]

X = data.drop(columns=[TARGET])
y_true = data[TARGET]

missing_features = [
    col for col in expected_features if col not in X.columns
]

if missing_features:
    st.error(f"Missing required features: {missing_features}")
    st.stop()

# Extra columns are ignored; required features are kept in training order.
X = X[expected_features]

# Check that uploaded labels match the model's known target classes.
try:
    if not set(y_true.unique()).issubset({0, 1}):
        st.error("Target values must be 0 or 1 for this dataset.")
        st.stop()
except TypeError:
    st.error("The target column contains invalid values.")
    st.stop()

if y_true.nunique() != 2:
    st.error(
        "The uploaded test data must contain both target classes "
        "to calculate ROC AUC reliably."
    )
    st.stop()

# --------------------------------------------------
# Metric calculation
# --------------------------------------------------
def evaluate_model(model, features, labels):
    predictions = model.predict(features)

    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(labels, predictions),
    }

    try:
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)

        # In this dataset, target 1 represents benign and target 0 malignant.
        positive_index = classes.index(1)
        metrics["AUC"] = roc_auc_score(
            labels, probabilities[:, positive_index]
        )
    except (AttributeError, ValueError, IndexError):
        metrics["AUC"] = float("nan")

    return metrics, predictions


# --------------------------------------------------
# Dataset overview
# --------------------------------------------------
st.markdown("---")
st.header("1. Test Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Test Instances", f"{len(data):,}")
col2.metric("Input Features", f"{X.shape[1]}")
col3.metric("Target Classes", f"{y_true.nunique()}")

with st.expander("Preview uploaded dataset"):
    st.dataframe(data.head(15), use_container_width=True)

# --------------------------------------------------
# Load saved models and compare all five
# --------------------------------------------------
all_metrics = {}
all_predictions = {}

for model_name, filename in MODEL_FILES.items():
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        st.error(
            f"Saved model not found: {model_path}. "
            "Run `python train_models.py` first."
        )
        st.stop()

    try:
        model = joblib.load(model_path)
        metrics, predictions = evaluate_model(model, X, y_true)
    except Exception as exc:
        st.error(f"Error evaluating {model_name}: {exc}")
        st.stop()

    all_metrics[model_name] = metrics
    all_predictions[model_name] = predictions

# --------------------------------------------------
# Comparison table
# --------------------------------------------------
st.markdown("---")
st.header("2. Comparison of All Five Models")

comparison = pd.DataFrame(all_metrics).T

metric_order = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC",
]

comparison = comparison[metric_order]

st.dataframe(
    comparison.style.format("{:.4f}"),
    use_container_width=True,
)

best_model = comparison["F1 Score"].idxmax()

st.success(
    f"Best model by weighted F1 on this test dataset: "
    f"{best_model} ({comparison.loc[best_model, 'F1 Score']:.4f})"
)

st.download_button(
    "Export comparison metrics as CSV",
    data=comparison.to_csv(index=True).encode("utf-8"),
    file_name="model_comparison_metrics.csv",
    mime="text/csv",
)

# --------------------------------------------------
# Detailed evaluation for selected model
# --------------------------------------------------
st.markdown("---")
st.header(f"3. Detailed Evaluation: {selected_model}")

selected_metrics = all_metrics[selected_model]
selected_predictions = all_predictions[selected_model]

metric_cols = st.columns(6)

for col, metric_name in zip(
    metric_cols,
    ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
):
    value = selected_metrics[metric_name]

    col.metric(
        metric_name,
        "N/A" if pd.isna(value) else f"{value:.4f}",
    )

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")

    labels = [0, 1]

    cm = confusion_matrix(
        y_true,
        selected_predictions,
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"],
        xlabel="Predicted Class",
        ylabel="Actual Class",
        title="Actual vs Predicted",
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# Classification report
# --------------------------------------------------
with right:
    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        selected_predictions,
        labels=[0, 1],
        target_names=["Malignant", "Benign"],
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.style.format("{:.4f}"),
        use_container_width=True,
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Machine Learning Assignment 2 | BITS Pilani WILP M.Tech AIML"
)
st.caption(
    "Evaluation metrics are calculated on the uploaded labelled test data. "
    "For valid evaluation, use test data that was not used to train the models."
)