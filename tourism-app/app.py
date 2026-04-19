import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# loading model from Hugging Face
model_path = hf_hub_download(repo_id="Dharini95/tourism-model", filename="best_model.pkl")

model = joblib.load(model_path)

# UI
st.title("Tourism Package Prediction App")

st.write("Enter the following customer details:")

# numeric inputs
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", 100)
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", 1)
NumberOfFollowups = st.number_input("Number of Followups", 1)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips", 1)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1)
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Children Visiting", 0)
MonthlyIncome = st.number_input("Monthly Income", 20000)

#categorical inputs
TypeOfContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product", ["Standard", "Deluxe", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Mariral Status", ["Single", "Married", "Unmarried"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP"])

# Getting input and saving them in base df
input_data = pd.DataFrame({
    'Age': [Age],
    'CityTier': [CityTier],
    'DurationOfPitch': [DurationOfPitch],
    'NumberOfPersonVisiting': [NumberOfPersonVisiting],
    'NumberOfFollowups': [NumberOfFollowups],
    'PreferredPropertyStar': [PreferredPropertyStar],
    'NumberOfTrips': [NumberOfTrips],
    'Passport': [Passport],
    'PitchSatisfactionScore': [PitchSatisfactionScore],
    'OwnCar': [OwnCar],
    'NumberOfChildrenVisiting': [NumberOfChildrenVisiting],
    'MonthlyIncome': [MonthlyIncome]
})

# adding dummy columns
dummy_cols = ['TypeofContact_Self Enquiry', 'Occupation_Large Business',
       'Occupation_Salaried', 'Occupation_Small Business', 'Gender_Female',
       'Gender_Male', 'ProductPitched_Deluxe', 'ProductPitched_King',
       'ProductPitched_Standard', 'ProductPitched_Super Deluxe',
       'MaritalStatus_Married', 'MaritalStatus_Single',
       'MaritalStatus_Unmarried', 'Designation_Executive',
       'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP']

for col in dummy_cols:
    input_data[col] = 0

# setting selected category as 1

# for TypeOfContact
if TypeOfContact == "Self Enquiry":
    input_data['TypeofContact_Self Enquiry'] = 1

# Occupation
input_data[f'Occupation_{Occupation}'] = 1

# Gender
input_data[f'Gender_{Gender}'] = 1

# Product
input_data[f'ProductPitched_{ProductPitched}'] = 1

# Marital Status
input_data[f'MaritalStatus_{MaritalStatus}'] = 1

# Designation
input_data[f'Designation_{Designation}'] = 1

# final column
final_columns = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
       'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
       'Passport', 'PitchSatisfactionScore', 'OwnCar',
       'NumberOfChildrenVisiting', 'MonthlyIncome',
       'TypeofContact_Self Enquiry', 'Occupation_Large Business',
       'Occupation_Salaried', 'Occupation_Small Business', 'Gender_Female',
       'Gender_Male', 'ProductPitched_Deluxe', 'ProductPitched_King',
       'ProductPitched_Standard', 'ProductPitched_Super Deluxe',
       'MaritalStatus_Married', 'MaritalStatus_Single',
       'MaritalStatus_Unmarried', 'Designation_Executive',
       'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP']

input_data = input_data[final_columns]

# prediction
if st.button("Predict"):
  prediction = model.predict(input_data)

  if prediction[0] == 1:
    st.success("Customer will purchase the Package")
  else:
    st.error("Customer will not purchase the Package")
