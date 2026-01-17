import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# ---------------- LOAD MODEL & FILES ----------------
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;color:#4CAF50;'>📊 Customer Churn Prediction App</h1>
    <p style='text-align:center;'>Predict whether a customer is likely to leave the bank</p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🧾 Customer Details")

geography = st.sidebar.selectbox(
    '🌍 Geography', onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    '👤 Gender', label_encoder_gender.classes_
)

age = st.sidebar.slider('🎂 Age', 18, 92, 30)

tenure = st.sidebar.slider('📆 Tenure (Years)', 0, 10, 3)

num_of_products = st.sidebar.slider(
    '📦 Number of Products', 1, 4, 1
)

has_cr_card = st.sidebar.selectbox(
    '💳 Has Credit Card', ['No', 'Yes']
)

is_active_member = st.sidebar.selectbox(
    '⚡ Is Active Member', ['No', 'Yes']
)

credit_score = st.sidebar.number_input(
    '🏦 Credit Score', min_value=300, max_value=900, value=650
)

balance = st.sidebar.number_input(
    '💰 Account Balance', value=50000.0
)

estimated_salary = st.sidebar.number_input(
    '💵 Estimated Salary', value=60000.0
)

# Convert Yes/No to 0/1
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# ---------------- MAIN SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Customer Summary")
    st.write(f"**Geography:** {geography}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Age:** {age}")
    st.write(f"**Tenure:** {tenure} years")
    st.write(f"**Products:** {num_of_products}")

with col2:
    st.subheader("📌 Financial Details")
    st.write(f"**Credit Score:** {credit_score}")
    st.write(f"**Balance:** ₹ {balance:,.2f}")
    st.write(f"**Estimated Salary:** ₹ {estimated_salary:,.2f}")
    st.write(f"**Credit Card:** {'Yes' if has_cr_card else 'No'}")
    st.write(f"**Active Member:** {'Yes' if is_active_member else 'No'}")

st.divider()

# ---------------- PREPARE INPUT ----------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

input_data_scaled = scaler.transform(input_data)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Churn", use_container_width=True):
    with st.spinner("Analyzing customer data..."):
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

    st.divider()
    st.subheader("📈 Prediction Result")

    st.progress(int(prediction_proba * 100))

    st.metric(
        label="Churn Probability",
        value=f"{prediction_proba:.2%}"
    )

    if prediction_proba > 0.5:
        st.error("❌ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
 
