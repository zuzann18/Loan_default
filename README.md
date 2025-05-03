

# Loan Default Prediction: End-to-End ML Pipeline

This project provides a comprehensive machine learning solution for predicting loan defaults, using the HMEQ (Home Equity) dataset. The goal is to help banks and financial institutions automate, optimize, and explain credit risk decisions, replacing manual, error-prone processes with robust, interpretable ML models.

---

## Project Overview

- **Business Problem:**  
  Loan defaults threaten bank profitability and stability. The manual approval process is slow and can introduce bias. Automating this decision with data science ensures fairness, transparency, and efficiency.
  
- **Objective:**  
  Build and interpret a classification model to identify loan applicants at risk of default, providing actionable insights for both business users and regulators.

---

## Features

- **Data Preprocessing:**  
  - Imputation of missing values  
  - Outlier detection and treatment  
  - Feature encoding and scaling  
  - Handling class imbalance (SMOTE)

- **Exploratory Data Analysis (EDA):**  
  - Univariate, bivariate, and multivariate analysis  
  - Visualization of feature distributions, relationships, and correlations

- **Model Development:**  
  - Logistic Regression (baseline, interpretable)
  - Random Forest (ensemble, non-linear)
  - XGBoost (state-of-the-art boosting)
  - Hyperparameter tuning with GridSearchCV

- **Evaluation:**  
  - Cross-validated model comparison (ROC-AUC, F1, precision, recall, confusion matrices)
  - Statistical tests to ensure robust model selection

- **Interpretability:**  
  - SHAP (SHapley Additive exPlanations) for feature importance and transparent decisions
  - Business-oriented recommendations for credit policy

- **Deployment-Readiness:**  
  - Save/load full preprocessing + model pipeline  
  - Notebook-based documentation suitable for stakeholders

---

## Results

- **Best Model:** XGBoost (ROC-AUC ≈ 0.96, high F1-score, best at reducing false negatives)
- **Key Features:** Debt-to-Income Ratio, Age of Oldest Credit Line, Delinquencies, Major Derogatory Reports
- **Interpretability:** SHAP analysis supports explainable AI for regulatory compliance and customer transparency

---

## Repository Contents

- `Applied_Data_Science_Loan_Default_Zuzanna_Walus.ipynb` – Full analysis and code (Jupyter notebook)
- `hmeq.csv` – Source data
- `Loan Default Prediction Problem Statement (1).pdf` – Original business statement
- `loancriteria.txt` – Project criteria and evaluation rubric

---

## How to Use

1. Clone the repository and open the main notebook in Jupyter or Colab
2. Follow the workflow from EDA to final model selection and interpretation
3. Adapt the code for your own credit risk or binary classification tasks

---

## Tags

`machine-learning` `loan-default` `credit-risk` `classification` `xgboost` `random-forest` `logistic-regression` `data-science` `interpretable-ml` `shap` `banking` `python` `sklearn` `imbalanced-data` `eda`

---

## Author

**Zuzanna Walus**

---

## License

For academic and non-commercial use only. Dataset and intellectual property © Great Learning / UCI.

---

## References

- [UCI ML Repository – HMEQ Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- SHAP documentation: https://shap.readthedocs.io

---
TODO
1) Features in the shap on the bar plot
2) mice and preprocsessing new bug with smote
   ---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[155], line 3
      1 # Apply SMOTE only on training data
      2 smote = SMOTE(random_state=42)
----> 3 X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
      4 # Plot class distribution after SMOTE
      5 plt.figure(figsize=(6,4))

File c:\Users\zuzan\AppData\Local\Programs\Python\Python311\Lib\site-packages\imblearn\base.py:208, in BaseSampler.fit_resample(self, X, y)
    187 """Resample the dataset.
    188 
    189 Parameters
   (...)
    205     The corresponding label of `X_resampled`.
    206 """
    207 self._validate_params()
--> 208 return super().fit_resample(X, y)

File c:\Users\zuzan\AppData\Local\Programs\Python\Python311\Lib\site-packages\imblearn\base.py:106, in SamplerMixin.fit_resample(self, X, y)
    104 check_classification_targets(y)
    105 arrays_transformer = ArraysTransformer(X, y)

- 



