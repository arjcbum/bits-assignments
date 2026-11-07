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
