# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 06:49:21 2023

@author: vaibhav
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('C:/Users/vaibhav/Desktop/diabetes_mini_proj/trained_model.sav', 'rb'))

# Create a function for prediction
def diabetes_prediction(input_data):
    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make the prediction
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return "The Person IS NOT DIABETIC"
    else:
        return "The Person IS DIABETIC"

def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')
    
    # Instructions
    st.write("Enter the following details to predict if a person has diabetes:")

    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    # Input validation
    if st.button('Diabetes Test Result'):
        if Pregnancies and Glucose and BloodPressure and SkinThickness and Insulin and BMI and DiabetesPedigreeFunction and Age:
            try:
                # Ensure the input values are in the correct format
                input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
                diagnosis = diabetes_prediction(input_data)
                st.success(diagnosis)
            except ValueError:
                st.error("Please enter valid numerical values for all fields.")
        else:
            st.error("Please fill in all the fields.")

if __name__ == '__main__':
    main()