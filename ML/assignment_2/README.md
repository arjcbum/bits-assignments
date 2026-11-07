# Classifier Lab — BITS Pilani M.Tech AIML, Machine Learning Assignment 2

## a. Problem statement
Implement five classification algorithms on one common dataset, compare them using six evaluation metrics, and demonstrate the saved models through an interactive Streamlit application. The app accepts a labelled test CSV, loads pre-trained models, calculates metrics, and displays a confusion matrix and classification report without fitting models on uploaded test data.

## b. Dataset description
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), distributed with scikit-learn and originally hosted by the University of Wisconsin / UCI Machine Learning Repository.

- **Task:** Binary classification.
- **Instances:** 569.
- **Input features:** 30 numeric measurements computed from digitized images of fine needle aspirates of breast masses.
- **Target:** `diagnosis` (0 = malignant, 1 = benign in the scikit-learn version).
- **Assignment minimums:** 12 features and 500 instances; both are satisfied.
- **Split:** 75% training / 25% held-out testing, stratified, `random_state=42` (426 training rows and 143 test rows).
- **Preprocessing:** `StandardScaler` inside pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. Decision Tree is trained without scaling.
- **Dataset source:** UCI Machine Learning Repository, Wisconsin Diagnostic Breast Cancer: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic . The project loads the public dataset using `sklearn.datasets.load_breast_cancer`.

## c. GitHub Repository Link
`https://github.com/arjcbum/bits-assignments/tree/main/ML/assignment_2`

The repository contains `app.py`, `train_models.py`, `requirements.txt`, `README.md`, `test_data.csv`, five saved models under `model/`, and experiment outputs under `outputs/`.

## d. Models used

### Evaluation comparison
All scores below are computed on the same stratified 25% held-out test partition. The positive class for precision, recall and F1 is class `1` (benign).

| Model                  |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9860 | 0.9977 |      0.9889 |   0.9889 | 0.9889 | 0.9700 |
| Decision Tree          |     0.9371 | 0.9186 |      0.9551 |   0.9444 | 0.9497 | 0.8657 |
| K-Nearest Neighbors    |     0.9790 | 0.9923 |      0.9677 |   1.0000 | 0.9836 | 0.9555 |
| Gaussian Naive Bayes   |     0.9371 | 0.9878 |      0.9355 |   0.9667 | 0.9508 | 0.8644 |
| Support Vector Machine |     0.9790 | 0.9969 |      0.9888 |   0.9778 | 0.9832 | 0.9553 |

### Observations on model performance
- **Logistic Regression:** Achieved the strongest overall result on this held-out split, with the highest accuracy, AUC and F1. Standardization helps the optimizer handle differently scaled measurements.
- **Decision Tree:** Had the lowest AUC and accuracy among these five models. The tree captures nonlinear splits and remains relatively interpretable, but its performance is weaker on this split.
- **K-Nearest Neighbors:** Reached perfect recall for the benign positive class on this test split, with strong F1 and AUC. Its distance calculations depend on standardized features.
- **Gaussian Naive Bayes:** Produced a comparatively high AUC despite lower accuracy. Its conditional-independence and Gaussian assumptions may be restrictive for correlated measurements.
- **Support Vector Machine:** Delivered strong accuracy, AUC and F1, close to KNN. The RBF kernel can capture nonlinear boundaries; feature standardization is important.

### Overall winner for this dataset
**Logistic Regression** is the winner under the primary comparison criterion of F1 score (ties are broken by AUC and then MCC). This is a result for this dataset and this particular split, not a universal ranking of classifiers.

## Streamlit application functionality
1. Dataset upload option for a labelled CSV.
2. Model-selection dropdown.
3. Accuracy, AUC, precision, recall, F1 and MCC display.
4. Confusion matrix and classification report.
5. Side-by-side comparison of all five saved models on the same uploaded test data.
6. Downloadable evaluation-results CSV.

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
streamlit run app.py
```

To regenerate the saved classifiers and training comparison:
```bash
python3 train_models.py
```

## Deployment
Deploy `app.py` from this repository using Streamlit Community Cloud. 
`https://bits-assignments-ml-asg2-2026aj05727.streamlit.app`