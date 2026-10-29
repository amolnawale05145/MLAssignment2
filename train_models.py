from pathlib import Path
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
(ROOT / "model").mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)

dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pd.concat([X_train, y_train.rename("target")], axis=1).to_csv(
    ROOT / "data/train_data.csv", index=False
)
pd.concat([X_test, y_test.rename("target")], axis=1).to_csv(
    ROOT / "test_data.csv", index=False
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "naive_bayes": GaussianNB(),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ])
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, ROOT / "model" / f"{name}.joblib")
    print(f"Saved model: {name}")

print(f"Features: {X.shape[1]}")
print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print("Dataset target mapping:", dict(enumerate(dataset.target_names)))
print("Training completed successfully.")