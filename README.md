
# 📘 ML Classifier Web App – Bank Marketing Dataset  
**Author:** Chinmay Das (2025AA05677)  
**Course:** Machine Learning  

---

## 🔗 Live Demo

- **Streamlit App:** https://bank-marketing-project-9rxqqtmagcw8thcd8ygwfm.streamlit.app  
- **GitHub Repository:** https://github.com/cd0130/bank-marketing-project  

---

# 🅐 Problem Statement

The goal of this project is to build and evaluate multiple Machine Learning models to predict whether a bank customer will **subscribe to a term deposit**. The prediction is based on customer demographic and past marketing interaction features from the **UCI Bank Marketing Dataset**.

A Streamlit web app is developed to allow users to:

- Upload a **test dataset** (CSV)  
- Select a **pre‑trained ML model**  
- View **evaluation metrics**  
- Inspect **confusion matrix** and **classification report** 

Model training is done offline in a Jupyter notebook.  
The Streamlit application performs **inference only**.

---

# 🅑 Dataset Description

The dataset used is the **UCI Bank Marketing Dataset**, sourced from direct marketing phone campaigns conducted by a Portuguese bank.

### Key Details:
- **Rows:** ~45,000  
- **Separator:** `;`  
- **Target Variable:** `y` (mapped: yes → 1, no → 0)  
- **Features:** age, job, marital status, education, balance, housing loan, campaign statistics, etc.  
- **Important Note:** The `duration` column is removed during training because it introduces **data leakage** (it is only known after the call ends).  

A small **balanced test file (~200 rows)** is included in `data/test_sample.csv` for Streamlit demo usage (due to free‑tier resource limits).

---

# 🅒 Models Used & Comparison Table

The following **six models** were trained:

- Logistic Regression  
- Decision Tree  
- K‑Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

---

## 📊 Comparison Table (Evaluation Metrics for All 6 Models)

| **ML Model** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
|--------------|-------------:|--------:|--------------:|-----------:|-------:|--------:|
| **XGBoost (Ensemble)**         | 0.8122 | 0.8017 | 0.6429 | 0.7379 | 0.6655 | 0.3687 |
| **Random Forest (Ensemble)**   | 0.8293 | 0.7983 | 0.6507 | 0.7266 | 0.6736 | 0.3696 |
| **Logistic Regression**        | 0.7548 | 0.7722 | 0.6028 | 0.6980 | 0.6104 | 0.2853 |
| **Naive Bayes**                | 0.8410 | 0.7558 | 0.6458 | 0.6841 | 0.6608 | 0.3277 |
| **KNN**                        | 0.8874 | 0.7027 | 0.7275 | 0.5882 | 0.6150 | 0.2833 |
| **Decision Tree**              | 0.8405 | 0.6071 | 0.6102 | 0.6071 | 0.6086 | 0.2174 |

---

# 🅓 Observations on Model Performance

| **ML Model** | **Observation about model performance** |
|--------------|------------------------------------------|
| **Logistic Regression** | Performs well as a baseline. Good AUC (0.77). Handles linear patterns effectively. Struggles with non‑linear boundaries. |
| **Decision Tree** | Simple and interpretable but prone to overfitting. Lowest AUC (0.61). Weak generalization. |
| **KNN** | Highest accuracy (0.8874) but lower recall. Sensitive to scaling and class imbalance; tends to miss positive cases. |
| **Naive Bayes** | Consistent and fast. Performs well despite strong assumptions. Good recall and balanced metrics. |
| **Random Forest (Ensemble)** | Stable, strong model. Excellent AUC (0.7983). Good precision/recall balance. Resistant to overfitting. |
| **XGBoost (Ensemble)** | **Highest AUC (0.8017)** and strong recall. Captures complex non‑linear relationships. Best overall balanced model. |

# Acknowledgements
Dataset Source: UCI Machine Learning Repository – Bank Marketing Dataset https://archive.ics.uci.edu/dataset/222/bank+marketing
Streamlit: https://streamlit.io
scikit‑learn: https://scikit-learn.org
XGBoost: https://xgboost.readthedocs.io
