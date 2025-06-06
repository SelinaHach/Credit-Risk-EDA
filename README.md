# Credit Risk Modeling Project

This project covers **end-to-end exploratory data analysis (EDA)**, **data cleaning**, and **modeling** using **Random Forest** and **XGBoost** to predict consumer credit charge-off risk based on anonymized credit bureau data for New-To-Credit (NTC) consumers.

---
##  Modeling
### A. Random Forest
- Used to predict `Ever_ChargeOff`.
- Trained with `class_weight='balanced'` to handle imbalance.
- Visualized a shallow `DecisionTreeClassifier` to interpret splits.
- Validation:
  - ROC AUC, classification report, confusion matrix.
  - Custom decision threshold (0.3) applied.
- Feature importance chart (top 15 features).

### B. XGBoost
- Trained using class-weighted configuration (`scale_pos_weight`) to address imbalance.
- Early stopping via validation set.
- Evaluation:
  - ROC curve and AUC for both validation and test sets.
  - Feature importance plotted and printed.
## Quick Summary: Random Forest vs. XGBoost for Credit Risk
- **Random Forest** is faster, easier to tune, and more interpretable. Ideal for quick baselines and when transparency is important.
- **XGBoost** usually delivers better recall and AUC, especially for imbalanced data like credit charge-off. However, it requires more tuning and is less interpretable.

Use **Random Forest** if you need speed, simplicity, or model transparency.  
Use **XGBoost** if your priority is **catching more charge-offs**, even at the cost of complexity and training time.



