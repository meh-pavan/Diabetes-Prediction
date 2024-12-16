# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 00:25:00 2024

@author: PAVAN
"""

import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np

diabetes_prediction_model = pickle.load(open('diabetes_prediction.sav','rb'))

st.title('Diabetes Prediction')

Pregnancies = st.text_input('Enter Number of Pregnancies')

Glucose = st.text_input('Enter Your Glucose Level Value')

BloodPressure = st.text_input('Enter Your Blood Pressure Value')

SkinThickness = st.text_input('Enter Your Skin Thickness Value')

Insulin = st.text_input('Enter Your Insulin Value')

BMI = st.text_input('Enter Your BMI Value')

DiabetesPedigreeFunction = st.text_input('Enter Your Diabetes Pedigree Function Value')

Age = st.text_input('Enter Your Age')

Diabetes_diagnosis= ''

scaler = StandardScaler()

if st.button('Check If Your Diabetic'):

    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    user_input = [float(x) for x in user_input]

    user_input_numpy_array = np.asarray(user_input)    
    
    reshaped_data = user_input_numpy_array.reshape(1,-1)
    
    standardized_data = scaler.fit_transform(reshaped_data)

    Diabetes_prediction = diabetes_prediction_model.predict(standardized_data)

    if Diabetes_prediction[0] == 1:
        Diabetes_diagnosis = "You are Diabetic"
    else:
        Diabetes_diagnosis = "Congrats! You are Healthy"

st.success(Diabetes_diagnosis)
