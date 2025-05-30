# Credit-Risk-EDA
Overview

This project performs a comprehensive exploratory data analysis (EDA) and machine learning modeling pipeline to predict consumer credit risk. We leverage a credit profile dataset to explore key features, detect and remove outliers, handle missing values, split the data into training, validation, and testing sets, and build both Random Forest and XGBoost classifiers.
Data Loading and Cleaning

Mount Drive (Colab): Mount Google Drive to access data in /content/sample_data/.

Load Data: Read the CSV file into a pandas DataFrame.

Drop Unnecessary Columns: Remove ID and Unpaid_CO_84M if present.

Outlier Detection & Removal:

Use IQR method on non-binary numerical columns.

Exclude target variables and columns with constrained ranges.

Missing Value Handling:

Identify missing values per column.

Impute missing FICO_V values with a special placeholder (e.g., 999).

Exploratory Data Analysis (EDA)

Display first few rows and DataFrame info.

Compute summary statistics for numerical features.

Value counts for categorical/binary columns.

Correlation analysis and heatmaps for target variables Ever_ChargeOff and ChargeOff_Balance.

Plot distributions (histograms + KDE) for numerical columns.

Identify and report potential outliers.

Data Splitting

Split the cleaned DataFrame into three sets:

Training: 70%

Validation: 15%

Test: 15%

Use train_test_split with fixed random_state=42 and stratify on the target variable where appropriate.

Modeling

Random Forest Classifier

Parameters:

n_estimators=100

max_depth=6

class_weight='balanced'

random_state=42

Train on the training set.

Evaluate on validation set with a custom threshold of 0.3 for Ever_ChargeOff probability.

Metrics: confusion matrix, classification report, ROC AUC.

Visualize a shallow decision tree (max_depth=3) for interpretability.

Plot top 15 feature importances.

XGBoost Classifier

Parameters:

objective='binary:logistic'

n_estimators=200

max_depth=5

learning_rate=0.1

scale_pos_weight calculated from training class imbalance

use_label_encoder=False

eval_metric='auc'

early_stopping_rounds=10

Fit with early stopping on the validation set.

Evaluate performance on validation and test sets.

Plot ROC curves and feature importances.
