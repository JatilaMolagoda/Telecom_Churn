# Telecom_Churn
An end to end Machine Learning pipeline for Telecom Customer Churn Prediction, evaluating LR, RF, and XGBoost with a business focus on optimizing Recall to minimize revenue loss
# 📞 Telecom Customer Churn Prediction & Analytics

A comprehensive, business-centric Machine Learning pipeline designed to proactively identify customers at risk of leaving a telecom provider (churning). This project focuses on solving the data science trade-off between standard statistical metrics and real-world business value.

---

## Executive Summary & Business Impact
In the telecom industry, acquiring a new customer is 5x more expensive than retaining an existing one.
* **The Goal:**
* Build a predictive pipeline to flag at-risk customers so marketing teams can offer proactive incentives.
* **The Solution:**
* A machine learning engine that captures **80% of churning customers (Recall: 0.80)**, allowing the business to mitigate revenue loss by taking targeted retention actions.

---

## The Data Science Trade-off (Why Accuracy Lies)

Standard baseline models often prioritize **Accuracy**, which can be highly misleading in imbalanced datasets. Here is how our models performed on the test set:

| Model | Accuracy | Recall (Churn Class) | F1-Score (Churn Class) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **81%** | 57% | 0.61 |
| **Random Forest (Final Selection)** | 67% | **80%** | **0.56** |
| **XGBoost** | 75% | 20% | 0.29 |

### Business Decision:
While Logistic Regression achieved the highest overall Accuracy (81%), it failed to detect 43% of actual churners. In contrast, the **Random Forest** model was selected because of its **80% Recall**. For a telecom provider, the cost of missing a churning customer is vastly higher than accidentally offering a retention discount to a loyal customer.

---

## Tech Stack & Architecture
* **Language:** Python 3
* **Environment:** Google Colab
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
* **Core Pipeline Steps:**
1. **EDA & Visualization:** Analyzed contract terms and billing metrics.
2. **Data Cleaning:** Handled missing/empty string objects in `TotalCharges`.
3. **Feature Encoding:** Converted categorical traits using One-Hot Encoding (`pd.get_dummies`).
4. **Stratified Train-Test Split:** Managed class distribution using an 80/20 partition.
5. **Model Evaluation:** Evaluated via Precision, Recall, and Confusion Matrices.

---

## How to Explore the Notebook
1. Open the `notebook.ipynb` file directly in GitHub or upload it to Google Colab.
2. Ensure you have the dataset from Kaggle (`Telco-Customer-Churn.csv`).
3. Run the cells sequentially to reproduce the data preprocessing, training, and comparative visualization steps.

---
Developed with by Jatila Molagoda. Connect with me on [LinkedIn]((https://www.linkedin.com/in/jatila-molagoda-9a330aa7/)).
