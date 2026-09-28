"""Streamlit web app for the disease prediction model.
Run:  streamlit run app.py   (after running train_model.py once)
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Prediction", page_icon="🩺")
st.title("🩺 Breast Cancer Prediction")
st.caption("Random Forest model trained on the Wisconsin Diagnostic dataset. "
           "For learning purposes only, not for medical use.")

bundle = joblib.load("models/rf_model.joblib")
model, features = bundle["model"], bundle["features"]

st.subheader("Enter the top 10 most important measurements")
st.write("Other features default to the dataset average.")

values = dict(bundle["means"])
cols = st.columns(2)
for i, feat in enumerate(bundle["top_features"]):
    lo, hi, mean = bundle["mins"][feat], bundle["maxs"][feat], bundle["means"][feat]
    values[feat] = cols[i % 2].slider(feat, float(lo), float(hi), float(mean))

if st.button("Predict"):
    row = pd.DataFrame([values])[features]
    prob = model.predict_proba(row)[0][1]
    if prob >= 0.5:
        st.error(f"Prediction: **Malignant** (probability {prob:.1%})")
    else:
        st.success(f"Prediction: **Benign** (malignant probability {prob:.1%})")
    st.progress(float(prob))
