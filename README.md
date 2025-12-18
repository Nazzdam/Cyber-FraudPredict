🔍 Project Overview
This project builds an end-to-end Fraud & Cyber-Threat Detection Engine using:
Machine Learning (LightGBM & Logistic Regression)
Behavioral Analytics
Market Stress Indicators
SHAP Explainability
Interactive Streamlit App

It predicts the probability that a given transaction is fraudulent or cyber-risky, combining user behaviour, transaction metadata, and macro-financial market conditions such as FX volatility, VIX levels, and repo rate shifts.

This project showcases skills in data science, FinTech analytics, cybersecurity modelling, machine learning engineering, and interactive dashboard development.


🧠 Key Features
✔️ Advanced Feature Engineering
Rolling user behaviour (1-hour & 24-hour transaction velocity)
Device/country mismatch
Z-score amount anomaly
Night-time transaction flag
Market stress composites (USD/ZAR returns, VIX spikes, repo rate changes)

✔️ Machine Learning Models
LightGBM (primary model)
Logistic Regression baseline
SMOTE optional for class imbalance
ROC-AUC & PR-AUC evaluations
KS-statistic analysis

✔️ Explainability
Global SHAP summary plot
Per-transaction SHAP waterfall explanation
Full transparency on why the model flags an event

✔️ Fraud Analytics (Visual Intelligence)
Heatmap: Device Type × Merchant Category
Heatmap: Country-Level Fraud Rates
Fraud distribution analysis
Market stress warnings

✔️ Interactive Streamlit App
Real-time fraud scoring
Risk tier badges (Low / Medium / High)
Dynamic SHAP explanations
Live market stress indicator
User-friendly web interface

🏗️ Project Architecture
fraud_cyber_prediction/
│── data/
│   ├── creditcard.csv
│   ├── transactions_with_features.csv
│── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│── models/
│   ├── preprocessor.joblib
│   ├── lgbm_model.joblib
│   ├── logreg_baseline.joblib
│── streamlit_app/
│   ├── app.py
│── requirements.txt
│── README.md
