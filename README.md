# Credit Risk Prediction using XGBoost

This project builds a binary classification model using XGBoost to predict whether a consumer will **ever charge off a loan** (`Ever_ChargeOff`) based on credit profile data. The workflow includes data preparation, model training, evaluation, and feature importance analysis.

## 📁 Project Structure

```
.
├── credit_risk_model.py   # Python script with full training + evaluation code
├── v_credit_data_NTC_v51825.csv  # Input dataset (replace with actual path)
└── README.md              # Project description and instructions
```

## 📊 Objective

To train and evaluate an XGBoost classifier for predicting **Ever_ChargeOff** with imbalanced class handling and performance metrics including ROC-AUC, precision, recall, and F1-score.

---

## 🛠️ Setup

### Requirements

- Python ≥ 3.7  
- Packages:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib

You can install them via:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

---

## 🚀 How to Run

1. **Prepare the Data:**

   Replace this line in the script with your cleaned dataset:

   ```python
   df = pd.read_csv('v_credit_data_NTC_v51825.csv')
   ```

   Perform cleaning if needed and assign the cleaned DataFrame to `df_cleaned`.

2. **Model Training and Evaluation:**

   The script automatically:
   - Splits the dataset into training, validation, and test sets.
   - Adjusts for class imbalance using `scale_pos_weight`.
   - Trains an `XGBClassifier` with early stopping on the validation set.
   - Evaluates on both validation and test sets using:
     - Confusion Matrix
     - Classification Report
     - ROC-AUC Score
     - ROC Curve plots

3. **Feature Importance:**

   After training, the top 15 most important features (by gain) are displayed and visualized with a horizontal bar chart.

---

## 📈 Output Examples

- **Validation/Test Performance:**

  Confusion matrix, precision/recall/F1, and AUC on both validation and test data.

- **ROC Curve:**

  Plots for both validation and test sets.

- **Feature Importance Plot:**

  A ranked bar chart showing the most impactful features in prediction.

---

## 🔍 Target and Feature Columns

- **Target Variable:** `Ever_ChargeOff`
- **Excluded Columns:** `'FICO_V'`, `'No_Hit'` (and the target column)

---

## 📌 Notes

- `early_stopping_rounds` is set to 10 to prevent overfitting.
- The model uses `'auc'` as the evaluation metric.
- Feature selection is manual based on domain knowledge—can be enhanced with correlation or mutual information.

---

## 📬 Contact

For questions, reach out to the project author or open an issue.
