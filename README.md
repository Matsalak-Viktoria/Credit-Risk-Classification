# Credit Risk Classification | [View Code](https://github.com/Matsalak-Viktoria/Credit-Risk-Classification/blob/main/Credit_Risk_Classification.ipynb)

## Overview
This project explores the implementation and evaluation of a machine learning pipeline for credit risk classification using the Credit Risk dataset.

The main goal of the project is not only to build classification models for predicting loan approval decisions, but also to compare the performance of Logistic Regression, Naive Bayes, Decision Tree, and K-Nearest Neighbors (KNN) using data preprocessing, outlier detection, and hyperparameter tuning.

The project focuses on the following prediction task:
- Credit Risk Classification - Predicting whether a loan application will be approved or rejected based on applicants' demographic, financial, and credit history information.

## Objectives
The main objectives of this project are:
- Perform Exploratory Data Analysis (EDA) to understand feature distributions and relationships with the target variable.
- Build and evaluate a machine learning pipeline for credit risk classification using the Credit Risk dataset.
- Analyze the experimental results by comparing the performance of Logistic Regression, Naive Bayes, Decision Tree, and K-Nearest Neighbors (KNN) to identify the most effective classification model.

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
The project workflow includes:
1. Exploratory Data Analysis (EDA)  
2. Logistic Regression Training and Evaluation
   - Preprocessing Pipeline Setup (for Outlier Detection)
   - Data Preprocessing (Imputation, Scaling)
   - Outlier Detection with Isolation Forest
   - Train/Test Split
   - Model Training Pipeline Setup
   - Cross-Validation with GridSearchCV
     - Data Preprocessing (Imputation, Encoding, Scaling)
     - Logistic Regression Model Training
     - Best Hyperparameter Selection
   - Prediction on Unseen Test Data
   - Model Evaluation
3. Naive Bayes Training and Evaluation
   - Same steps as for Logistic Regression
4. Decision Tree Training and Evaluation
   - Same steps as for Logistic Regression
5. KNN Training and Evaluation
   - Same steps as for Logistic Regression
6. Result Analysis
   - Comparison of Model Performance
   - Selection of the Most Effective Classification Model

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Methods
### Data Preprocessing
**For Outlier Detection**:
- Missing value imputation (Median)
- Feature scaling (StandardScaler)

**For Model Training**:

**Numerical features**:
- Missing value imputation (Median)
- Feature scaling (StandardScaler)

**Categorical features**:
- Missing value imputation (Most Frequent)
- One-Hot Encoding

### Outlier Detection
- Isolation Forest

### Machine Learning Models
- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)

**Hyperparameters optimized**:
- Logistic Regression: Regularization strength (C)
- Decision Tree: Maximum tree depth (max_depth)
- K-Nearest Neighbors (KNN): Number of neighbors (n_neighbors)

### Validation Strategy
**Train/Test Split + GridSearchCV**:
- Train/Test split for final model evaluation
- GridSearchCV with 5-Fold Cross-Validation for hyperparameter optimization

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

## Results
Evaluation on the test set:
              precision    recall  f1-score   support

    Rejected       0.86      0.54      0.67        35
    Approved       0.83      0.96      0.89        82

    accuracy                           0.84       117
   macro avg       0.85      0.75      0.78       117
weighted avg       0.84      0.84      0.83       117
