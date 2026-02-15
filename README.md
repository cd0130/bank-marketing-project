📘 ML Classifier Web App – Bank Marketing Dataset
Author: Chinmay Das (2025AA05677)
Course: Machine Learning

## Live Demo

- **Streamlit App:** https://bank-marketing-project-9rxqqtmagcw8thcd8ygwfm.streamlit.app  
- **GitHub Repository:** https://github.com/cd0130/bank-marketing-project


🔍 1. Project Overview
This project implements a Machine Learning classification system built using Python, scikit‑learn, and deployed as an interactive Streamlit web application.
The goal is to predict whether a customer will subscribe to a term deposit (target: y) using the UCI Bank Marketing Dataset.
The app evaluates multiple pre‑trained models on uploaded test data and provides:
✔ Evaluation metrics
✔ Confusion matrix
✔ Classification report
✔ Downloadable sample test data
The focus is on model inference, not training. Model training is performed offline in a separate notebook.

📂 2. Dataset Description
The dataset used in training is the UCI Bank Marketing Dataset (semicolon ; separated).
It contains information collected from direct marketing phone campaigns conducted by a Portuguese bank.
Key details

Rows: ~45,000
Target variable: y (yes/no → mapped to 1/0)
Features: Age, job, marital status, balance, loan status, contact type, campaign history, etc.
Leakage removal: duration feature is excluded from model training to prevent leakage.

Only a small sample test CSV (~200 rows) is included in this repository for Streamlit evaluation.

🛠️ 3. Models Used (Pre‑trained)
The following ML models were trained offline and saved as .joblib files:

Logistic Regression
Decision Tree Classifier
K‑Nearest Neighbors (KNN) (optional if file is small enough)
Naive Bayes
Random Forest (excluded from repo if >100MB)
XGBoost Classifier

Only small model artifacts are included in the repo to ensure Streamlit Cloud performance.

🌐 4. Streamlit App Features
The deployed web app includes:
✔ a. Upload Test Dataset (CSV)

Only test data is uploaded (no full training data).
CSV separator (; or ,) is auto‑detected.

✔ b. Model Selection Dropdown

Shows full model names (e.g., “Logistic Regression”).

✔ c. Evaluation Metrics Display
Includes:

Accuracy
AUC
Precision
Recall
F1‑Score
Matthews Correlation Coefficient (MCC)

✔ d. Confusion Matrix Visualization

Compact custom rendering
Helps understand class‑wise prediction performance

✔ e. Classification Report
Displayed at the bottom (expanded by default).
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
