import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load pickled files
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehotencoder_geo.pkl', 'rb') as file:
    onehotencoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit App
st.title('Customer Churn Prediction')

# User input widgets
geography = st.selectbox('Geography', onehotencoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('CreditScore', min_value=0)
estimated_salary = st.number_input('EstimatedSalary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('NumOfProducts', 1, 4)
has_cr_card = st.selectbox('HasCrCard', [0, 1])
is_active_member = st.selectbox('IsActiveMember', [0, 1])

# Convert Gender to encoded form
gender_encoded = label_encoder_gender.transform([gender])[0]

# Base input (without geography)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode geography
geo_encoded = onehotencoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehotencoder_geo.get_feature_names_out(['Geography'])
)

# Merge everything
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_probability = float(prediction[0][0])

st.subheader(f"Prediction Probability: **{prediction_probability:.4f}**")

# Output
if prediction_probability > 0.5:
    st.error("🚨 **Customer is likely to churn**")
else:
    st.success("✅ **Customer is not likely to churn**")
