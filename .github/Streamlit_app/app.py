import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Fraud & Cyber-Threat Prediction",
    page_icon="🚨",
    layout="wide")

#--------
#Load Model +processing
#--------

@st.cache_resource
def LoadArtifacts():
    pre=joblib.load("models/preprocessor.joblib")
    model=joblib.load("gbm_model.joblib")
    return pre, model

preprocessor, model=LoadArtifacts()

#SHAP Explainer
shap.initjs()
explainer = shap.TreeExplainer(model)

st.title("🚨 Fraud & Cyber-Threat Prediction App")
st.markdown("""
This tool predicts the probability of a transaction being fraudulent or cyber-risky  
based on transaction metadata, user behaviour, and market stress conditions.
""")

# -----------------------------
# USER INPUT FORM
# -----------------------------

st.sidebar.header("input transaction features")

Aomunt=st.sidebar.number_input("AmountZAR", 1.0, 1000000.0, 250.50)
Hours=st.sidebar.number_input("HourOfDay", 0, 23, 12)
Weekday=st.sidebar.number_input("DayOfWeeek, (0=Mon)", list(range(7)))
DeviceType=st.sidebar.selectbox("DevciceType", ["Mobile", "Desktop", "Tablet"])
MerchantCategory=st.sidebar.selectbox("MerchantCategory", 
                                 ["Electronics", "Clothing", "Groceries", "Travel", "Entertainment"])
Country=st.sidebar.selectbox("Transaction Country", ["USA", "UK", "ZA", "NG", "IN", "CN"])

#Behavioural Features 
st.sidebar.header("User Behaviour")
UserTx1H=st.sidebar.number_input("User transaction count (1h window)", 0, 50, 0)
UserTx24H=st.sidebar.number_input("User transaction count (24h window)", 0, 200, 0)
UserTx7D=st.sidebar.number_input("User transaction count (7d window)", 0, 1000, 0)
UserAmtMean=st.sidebar.number_input("User average transaction amount",1.0, 20000.0, 500.0)
AmtZscore=st.sidebar.number_input("Amount Z-Score", -5.0, 10.0, 0.5)

#Market Stress Features
st.sidebar.header("Market Stress Indicators")
UsdZarRet=st.sidebar.number_input("USDZAR daily return", -0.2, 0.2, 0.01, step=0.001)
UsdZarVol=st.sidebar.mumber_input("USDZAR 7d volatility", 0.0, 0.2, 0.03, step=0.001)
Vix=st.sidebar.number_input("VIX Index level", 5.0, 80.0, 18.0,step=1.0)
Vix7D=st.sidebar.number_input("VIX 7d mean", 5.0, 80.0, 16.0, step=1.0)
Repo=st.sidebar.number_input("Repo Rate (%)", 3.0, 12.0, 6.5, step=0.25)
StressFlag=st.sidebar.selectbox("Market Stress Flag", [0, 1])
IsNight=st.sidebar.selectbox("Is Night Transaction", [0, 1])
DeviceCountryMismatch=st.sidebar.selectbox("Device-Country Mismatch", [0, 1])
