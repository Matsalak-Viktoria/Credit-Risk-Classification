# Credit Risk Classification | [View Code](https://github.com/Matsalak-Viktoria/Credit-Risk-Classification/blob/main/Credit_Risk_Classification.ipynb)

## Overview
This project explores the implementation and evaluation of a machine learning pipeline for credit risk classification using the Credit Risk dataset.

The main goal of the project is not only to build classification models for predicting loan approval decisions, but also to compare the performance of Logistic Regression, Naive Bayes, Decision Tree, and K-Nearest Neighbors (KNN) using data preprocessing, outlier detection, and hyperparameter tuning.

The project focuses on the following prediction task:
- Credit Risk Classification - Predicting whether a loan application will be approved or rejected based on applicants' demographic, financial, and credit history information.

## Objectives
The main objectives of this project are:
- Perform Exploratory Data Analysis (EDA) to understand feature distributions and relationships with the target variable.
- Build and evaluate machine learning models for credit risk classification using the Credit Risk dataset.
- Compare the performance of Logistic Regression, Naive Bayes, Decision Tree, and K-Nearest Neighbors (KNN) to identify the most effective classification model.

## Dataset
- **Loan ID**: Unique identifier of the loan application in the database, assigned automatically by the system.
- **Gender**: Applicant's gender. Possible values: **Male**, **Female**.
- **Married**: Applicant's marital status. Possible values: **Yes**, **No**.
- **Dependents**: Number of the applicant's dependents. Possible values: **0**, **1**, **2**, **3+**.
- **Education**: Applicant's education level. Possible values: **Graduate**, **Not Graduate**.
- **Self_Employed**: Indicates whether the applicant is self-employed. Possible values: **Yes**, **No**.
- **ApplicantIncome**: Applicant's average monthly income (USD).
- **CoapplicantIncome**: Co-applicant's (spouse's) average monthly income (USD).
- **LoanAmount**: Requested loan amount (thousand USD).
- **Loan_Amount_Term**: Loan term (months).
- **Credit_History**: Indicates whether the applicant has a previous credit history. Possible values: **Yes**, **No**.
- **Property_Area**: Property location category. Possible values: **Urban**, **Semiurban**, **Rural**.
- **Loan_Status**: Final loan application decision. Possible values: **Y** (Approved), **N** (Rejected).

## Workflow

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Methods

## Results
Evaluation on the test set:
              precision    recall  f1-score   support

    Rejected       0.86      0.54      0.67        35
    Approved       0.83      0.96      0.89        82

    accuracy                           0.84       117
   macro avg       0.85      0.75      0.78       117
weighted avg       0.84      0.84      0.83       117
