# AIML Assignment 2 — Classification Model Comparison

## a. Problem statement
Implement and compare five supervised classification algorithms on one public dataset using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC). Demonstrate evaluation through an interactive Streamlit app.

## b. Dataset description
**Dataset:** Breast Cancer Wisconsin (Diagnostic), distributed through `sklearn.datasets.load_breast_cancer` and originating from the UCI Machine Learning Repository.

- Problem type: binary classification
- Total instances: 569
- Input features: 30 numeric features
- Target: `target` (0 = malignant, 1 = benign in scikit-learn's dataset)
- Split: stratified 75% training / 25% test, `random_state=42`
- Test rows: 143
- Preprocessing: StandardScaler inside pipelines for Logistic Regression, KNN and SVM. Tree and Gaussian Naive Bayes are trained without scaling.
- The uploaded test CSV is evaluated only; the Streamlit app does not retrain on uploaded test data.

Dataset source: UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## c. GitHub Repository Link
**Add your repository URL here after creating and pushing the repository:** `https://github.com/amolnawale05145/MLAssignment2.git`

## d. Models used

Models: Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM).

### Evaluation metrics

Metrics below were calculated on the stratified hold-out test set included in `test_data.csv`. Precision, Recall and F1 use weighted averaging. AUC is ROC AUC using predicted probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9860 | 0.9977 | 0.9860 | 0.9860 | 0.9860 | 0.9700 |
| Decision Tree | 0.9371 | 0.9186 | 0.9374 | 0.9371 | 0.9372 | 0.8657 |
| K-Nearest Neighbors | 0.9790 | 0.9845 | 0.9797 | 0.9790 | 0.9789 | 0.9555 |
| Gaussian Naive Bayes | 0.9371 | 0.9893 | 0.9382 | 0.9371 | 0.9364 | 0.8650 |
| Support Vector Machine | 0.9790 | 0.9969 | 0.9792 | 0.9790 | 0.9791 | 0.9553 |

### Observations about model performance

Use these as initial interpretations, then verify them by running the code in BITS Virtual Lab and writing the observations in your own words.

- **Logistic Regression:** It achieved the strongest accuracy and weighted F1 on this split. Its high AUC indicates strong separation of the two classes. Standardization helps because the features have different scales.
- **Decision Tree:** It is easy to interpret, but its performance is lower than the other models on this split. A single tree can be sensitive to the training sample and may not generalize as well.
- **K-Nearest Neighbors:** It performed strongly after feature scaling. KNN depends on distances, so scaling is important; its performance can vary with the choice of `n_neighbors`.
- **Gaussian Naive Bayes:** It achieved lower accuracy than Logistic Regression and SVM, but a high AUC. Its conditional-independence and Gaussian assumptions may not fully match the dataset.
- **Support Vector Machine:** It produced strong accuracy and AUC. Scaling is important for SVM; `probability=True` enables AUC evaluation from predicted probabilities.
- **Overall winner on this hold-out split:** Logistic Regression by weighted F1 (and accuracy). This is a dataset- and split-specific result, not a claim that it is always the best classifier.

### App features
- Upload a labelled CSV file with all required feature columns and `target`.
- Select a model for detailed inspection.
- Compare metrics for all five models on the same uploaded test data.
- View a confusion matrix and classification report.
- Download the comparison table as CSV.
- Models are loaded from `model/`; they are not retrained by the app.

## How to run locally / in BITS Virtual Lab

1. Create a Python environment and install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. (Optional, if regenerating the artifacts) run:
   ```bash
   python train_models.py
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```
4. Upload `test_data.csv` and compare the model results.
5. Take your own screenshot of the running project in BITS Virtual Lab for the assignment evidence.

## Repository structure

```text
ml_assignment2_project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
├── data/
│   └── train_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## Deployment checklist

1. Push this project to a GitHub repository you own. Review and personalize the code, UI, observations, and README.
2. In Streamlit Community Cloud, create a new app from your repository and select `app.py`.
3. After successful deployment, replace the placeholder repository URL above and add the live Streamlit app URL.
4. Include a screenshot you personally captured from BITS Virtual Lab in the final PDF.
5. Open the GitHub link and live app link in a private/incognito browser window to confirm they are accessible.

## Submission links (fill these in before exporting to PDF)
- GitHub repository: **https://github.com/amolnawale05145/MLAssignment2.git/**
- Live Streamlit app: **https://ml-classification-assignment.streamlit.app/**
- BITS Virtual Lab screenshot: **[INSERT YOUR OWN SCREENSHOT]**
