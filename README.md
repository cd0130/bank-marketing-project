
# 📘 ML Classifier Web App – Bank Marketing Dataset  
**Author:** Chinmay Das (2025AA05677)  
**Course:** Machine Learning  

---

## 🔗 Live Demo

- **Streamlit App:** https://bank-marketing-project-9rxqqtmagcw8thcd8ygwfm.streamlit.app  
- **GitHub Repository:** https://github.com/cd0130/bank-marketing-project  

---

# 🅐 Problem Statement

The goal of this project is to build and evaluate multiple Machine Learning models to predict whether a bank customer will **subscribe to a term deposit**.  
The prediction is based on customer demographic and past marketing interaction features from the **UCI Bank Marketing Dataset**.

A Streamlit web app is developed to allow users to:

- Upload a **test dataset** (CSV)  
- Select a **pre‑trained ML model**  
- View **evaluation metrics**  
- Inspect **confusion matrix** and **classification report**  
- Download predictions when no target column is present  

Model training is done offline in a Jupyter notebook.  
The Streamlit application performs **inference only**.

---

# 🅑 Dataset Description *(1 mark)*

The dataset used is the **UCI Bank Marketing Dataset**, sourced from direct marketing phone campaigns conducted by a Portuguese bank.

### Key Details:
- **Rows:** ~45,000  
- **Separator:** `;`  
- **Target Variable:** `y` (mapped: yes → 1, no → 0)  
- **Features:** age, job, marital status, education, balance, housing loan, campaign statistics, etc.  
- **Important Note:** The `duration` column is removed during training because it introduces **data leakage** (it is only known after the call ends).  

A small **balanced test file (~200 rows)** is included in `data/test_sample.csv` for Streamlit demo usage (due to free‑tier resource limits).

---

# 🅒 Models Used & Comparison Table *(6 marks)*  

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

# 🅓 Observations on Model Performance *(3 marks)*

| **ML Model** | **Observation about model performance** |
|--------------|------------------------------------------|
| **Logistic Regression** | Performs well as a baseline. Good AUC (0.77). Handles linear patterns effectively. Struggles with non‑linear boundaries. |
| **Decision Tree** | Simple and interpretable but prone to overfitting. Lowest AUC (0.61). Weak generalization. |
| **KNN** | Highest accuracy (0.8874) but lower recall. Sensitive to scaling and class imbalance; tends to miss positive cases. |
| **Naive Bayes** | Consistent and fast. Performs well despite strong assumptions. Good recall and balanced metrics. |
| **Random Forest (Ensemble)** | Stable, strong model. Excellent AUC (0.7983). Good precision/recall balance. Resistant to overfitting. |
| **XGBoost (Ensemble)** | **Highest AUC (0.8017)** and strong recall. Captures complex non‑linear relationships. Best overall balanced model. |

---

# 📘 Project Overview

This project applies various ML classifiers to predict term deposit subscription using Python and scikit-learn — deployed using Streamlit.

The app provides:

- Evaluation metrics  
- Confusion matrices  
- Classification reports  
- Clean UI to test multiple ML models  

---

# 📂 Repository Structure
Includes:

Precision
Recall
F1‑score
Support
Macro average
Weighted average
Overall accuracy


📦 5. Repository Structure
bank-marketing-project/
│
├── app.py                     # Streamlit app
├── requirements.txt           # Dependencies for Streamlit Cloud
├── README.md                  # Project documentation
│
├── data/
│   └── test_sample.csv        # Small test sample for Quick Download (200 rows)
│
└── model/
    ├── model_logreg.joblib
    ├── model_tree.joblib
    ├── model_nb.joblib
    ├── model_xgb.joblib
    └── feature_columns.json   # Encoded feature names

(Large models like Random Forest or KNN are NOT included due to GitHub & Streamlit limits.)

🚀 6. Deployment Instructions (Streamlit Cloud)

Push this repository to a public GitHub repo.
Visit https://share.streamlit.io
Click New App
Select:

Repository: your GitHub repo
Branch: main
App file: app.py


Click Deploy

The app will build automatically and give a shareable public URL.

▶️ 7. How to Run the App Locally
Create a virtual environment and install dependencies:
Shellpip install -r requirements.txtShow more lines
Then run:
Shellstreamlit run app.pyShow more lines

📑 8. How to Use the App

Download the sample test CSV from the app.
Upload your test CSV (same schema as the training data).
Choose a model from the dropdown.
View:

Evaluation metrics
Confusion matrix
Classification report


Optionally download prediction results when no target column is present.


🧠 9. Training Notebook (Offline)
All training, preprocessing, and model saving is done in:
model_building.ipynb

This notebook:

Preprocesses training data
Trains all models
Evaluates performance
Saves .joblib artifacts
Generates a small test CSV
Extracts encoded feature names


🏁 10. Acknowledgements
Dataset Source:
UCI Machine Learning Repository – Bank Marketing Dataset
https://archive.ics.uci.edu/dataset/222/bank+marketing
Streamlit: https://streamlit.io
scikit‑learn: https://scikit-learn.org
XGBoost: https://xgboost.readthedocs.io
