# -*- coding: utf-8 -*-

import pickle

import pandas as pd
import streamlit as st
import sklearn  # needed for pickle compatibility


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    with open("my_model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()


# -----------------------------
# App title
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'>
        <b>Home Equity Loan Approval</b>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.header("Enter Loan Applicant's Details")


# -----------------------------
# User inputs
# -----------------------------
loan = st.slider("Loan Amount (LOAN)", min_value=1000, max_value=500000, step=1000)
mortdue = st.slider("Mortgage Due (MORTDUE)", min_value=0.0, max_value=1000000.0, step=1000.0)
value = st.slider("Property Value (VALUE)", min_value=0.0, max_value=1000000.0, step=1000.0)
yoj = st.selectbox("Years at Job (YOJ)", options=list(range(1, 41)))
derog = st.number_input("Derogatory Reports (DEROG)", min_value=0, max_value=15, step=1)
delinq = st.selectbox("Delinquent Reports (DELINQ)", options=list(range(0, 15)))
clage = st.slider("Age of Oldest Trade Line in Months (CLAGE)", min_value=0.0, max_value=100.0, step=1.0)
ninq = st.slider("Number of Recent Credit Inquiries (NINQ)", min_value=0.0, max_value=15.0, step=1.0)
clno = st.slider("Number of Credit Lines (CLNO)", min_value=0.0, max_value=50.0, step=1.0)
debtinc = st.slider("Debt-to-Income Ratio (DEBTINC)", min_value=0.0, max_value=200.0, step=0.1)

reason = st.selectbox("Reason for Loan (REASON)", ["HomeImp", "DebtCon"])
job = st.selectbox("Job Category (JOB)", ["ProfExe", "Other", "Mgr", "Office", "Sales"])


# -----------------------------
# Create input DataFrame
# -----------------------------
input_data = pd.DataFrame(
    {
        "LOAN": [loan],
        "MORTDUE": [mortdue],
        "VALUE": [value],
        "YOJ": [yoj],
        "DEROG": [derog],
        "DELINQ": [delinq],
        "CLAGE": [clage],
        "NINQ": [ninq],
        "CLNO": [clno],
        "DEBTINC": [debtinc],
        "REASON": [reason],
        "JOB": [job],
    }
)


# -----------------------------
# Match model training columns
# -----------------------------
input_data_encoded = pd.get_dummies(input_data, columns=["REASON", "JOB"])

model_columns = model.feature_names_in_

for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

input_data_encoded = input_data_encoded[model_columns]


# -----------------------------
# Prediction
# -----------------------------
if st.button("Evaluate Loan"):
    prediction = model.predict(input_data_encoded)[0]

    if prediction == 1:
        st.error("The prediction is: Bad Loan 🚫")
    else:
        st.success("The prediction is: Good Loan 💲")
