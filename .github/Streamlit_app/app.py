import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

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
    model=joblib.load("models/lgbm_model.joblib")
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

Amount=st.sidebar.number_input("AmountZAR", 1.0, 1000000.0, 250.50)
Hours=st.sidebar.number_input("HourOfDay", 0, 23, 12)
Weekday=st.sidebar.number_input("DayOfWeek, (0=Mon)", min_value=0, max_value=6, value=0)
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
UsdZarVol=st.sidebar.number_input("USDZAR 7d volatility", 0.0, 0.2, 0.03, step=0.001)
Vix=st.sidebar.number_input("VIX Index level", 5.0, 80.0, 18.0,step=1.0)
Vix7D=st.sidebar.number_input("VIX 7d mean", 5.0, 80.0, 16.0, step=1.0)
Repo=st.sidebar.number_input("Repo Rate (%)", 3.0, 12.0, 6.5, step=0.25)
StressFlag=st.sidebar.selectbox("Market Stress Flag", [0, 1])
IsNight=st.sidebar.selectbox("Is Night Transaction", [0, 1])
DeviceCountryMismatch=st.sidebar.selectbox("Device-Country Mismatch", [0, 1])

# -----------------------------
# Construct a single-row dataframe
# -----------------------------

InputData=pd.DataFrame({
    'Amount': [Amount],
    'HourOfDay': [Hours],
    'DayOfWeek': [Weekday],
    'UserTx1H': [UserTx1H],
    'UserTx24H': [UserTx24H],
    'UserTx7D': [UserTx7D],
    'UserAmtMean': [UserAmtMean],
    'AmtZscore': [AmtZscore],
    
    'DeviceType': [DeviceType],
    'MerchantCategory': [MerchantCategory],
    'Country': [Country],
    
    'UsdZarRet': [UsdZarRet],
    'UsdZarVol': [UsdZarVol],
    'Vix': [Vix],
    'Vix7D': [Vix7D],
    'Repo': [Repo],
    'StressFlag': [StressFlag],
    'IsNight': [IsNight],
    'DeviceCountryMismatch': [DeviceCountryMismatch]
})

# -----------------------------
# Prediction button
# -----------------------------

st.subheader("🔍 Prediction Results")

if st.button("Run fraud results"):
    #preprocess
    XProc=preprocessor.transform(InputData)
    #predict
    Prob=model.predict_proba(XProc)[0][1]
    ProbPct=round(Prob*100,2)
  
  # Risk message
    if Prob < 0.05:
        risk = ("🟢 LOW RISK", "green")
    elif Prob < 0.20:
        risk = ("🟡 MEDIUM RISK", "orange")
    else:
        risk = ("🔴 HIGH RISK", "red")

    st.markdown(f"### Fraud Probability: **{ProbPct}%**")
    st.markdown(
        f"<h3 style='color:{risk[1]}; font-weight:bold;'>{risk[0]}</h3>",
        unsafe_allow_html=True
    )

#Market stress badge
    if StressFlag==1:
        st.mardwon("⚠️ **Market Stress Conditions Detected!** ⚠️")
    else:
        st.markdown("✅ **Normal Market Conditions** ✅")
        

    
    #-----------------------------
    #SHAP Explanation
    # -----------------------------
    st.markdown("### 🔎 Model Explanation")
    ShapVal=explainer.shap_values(XProc)

    fig, ax=plt.subplots(figsize=(10,4))
    shap.waterfall_plot=getattr(shap.plots, "watefrfall", None)
    shap.waterfall_plot(explainer.expected_value, ShapVal[0], feature_names=preprocessor.get_feature_names_out(), max_display=10, show=False)

    st.pyplot(fig)

else:
    st.info("⚠️ Click the 'Run fraud results' button to generate predictions based on the input features.")
    
# ------------------------------------------------------------------------------
# Fraud Heatmap: Device Type × Merchant Category
# ------------------------------------------------------------------------------
st.subheader("📊 Fraud Heatmap: Device Type × Merchant Category")
@st.cache_data
def LoadHetmapData():
   return pd.read_csv("/Users/milanichikeka/Downloads/my_dataset.csv")

try:
    DfFull=LoadHetmapData()
    HeatMapData=(
        DfFull.groupby(['DeviceType', 'MerchantCategory'])['IsFraud'].mean()
    .reset_index()
    .pivot(index='DeviceType', columns='MerchantCategory', values='IsFraud')
    )
    
    Fig2, ax=plt.subplots(figsize=(10,6))
    sns.heatmap(
        HeatMapData,
        annot=True,
        fmt=".2%",
        cmap="Reds",
        cbar_kws={'label': 'Fraud Rate'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title("Fraud Rate by Device Type and Merchant Category")
    st.pyplot(Fig2)
except Exception as e:
    st.info("⚠️ Unable to load heatmap data at this time.")
    st.error(f"Error details: {e}")
    
# ------------------------------------------------------------------------------
# Fraud Heatmap: Country
# ------------------------------------------------------------------------------
st.subheader("📊 Fraud Heatmap: Transaction Country")    
try:
    CountryHeatmapData=(
        DfFull.groupby('Country')['IsFraud'].mean().reset_index()
    )
    
    Fig3, ax=plt.subplots(figsize=(8,4))
    sns.heatmap(
        CountryHeatmapData.set_index('Country').T,
        annot=True,
        cmap="Blues",
        fmt=".3f",
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_ylabel("Fraud Rate by country")
    ax.set_title("Fraud Rate by Transaction Country")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    st.pyplot(Fig3)
except Exception as e:
    st.info("⚠️ Unable to load country heatmap data at this time.")
    st.error(f"Error details: {e}")