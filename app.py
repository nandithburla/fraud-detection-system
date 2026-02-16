import streamlit as st
import joblib
import pandas as pd

# Load model and scalers
model = joblib.load("fraud_detection_model.pkl")
amount_scaler = joblib.load("amount_scaler.pkl")
time_scaler = joblib.load("time_scaler.pkl")

columns = [
    "scaled_amount", "scaled_time",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"
]

st.title("💳 Fraud Detection System")
st.write("Enter transaction details to predict whether it is Fraud or Normal.")

amount = st.number_input("Transaction Amount", value=100.0)
time = st.number_input("Transaction Time (seconds)", value=50000.0)

# For demo simplicity, PCA fields default to 0
pca_values = [0.0]*28

if st.button("Predict"):
    amount_scaled = amount_scaler.transform([[amount]])[0][0]
    time_scaled = time_scaler.transform([[time]])[0][0]

    final_features = [amount_scaled, time_scaled] + pca_values
    final_df = pd.DataFrame([final_features], columns=columns)

    prediction = model.predict(final_df)[0]
    probability = model.predict_proba(final_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected (Probability: {probability:.4f})")
    else:
        st.success(f"✅ Normal Transaction (Probability: {probability:.4f})")
