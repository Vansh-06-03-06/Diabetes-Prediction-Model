import streamlit as st
import numpy as np
import pickle

# Load the saved model and scaler
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))

def diabetes_prediction(input_data):
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data
    std_data = loaded_scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return 'The person is **Not Diabetic**.'
    else:
        return 'The person is **Diabetic**.'

def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')
    st.write('Enter the medical details to predict whether the person is diabetic or not.')
    
    st.markdown("---")

    # Getting the input data from the user in columns
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)
        BloodPressure = st.number_input('Blood Pressure value (mm Hg)', min_value=0.0, max_value=150.0, value=70.0)
        Insulin = st.number_input('Insulin Level (IU/mL)', min_value=0.0, max_value=1000.0, value=79.0)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=3.0, value=0.47)
        
    with col2:
        Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0.0, max_value=300.0, value=120.0)
        SkinThickness = st.number_input('Skin Thickness value (mm)', min_value=0.0, max_value=100.0, value=20.0)
        BMI = st.number_input('BMI value', min_value=0.0, max_value=70.0, value=31.9)
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, value=33, step=1)

    # Code for Prediction
    diagnosis = ''
    
    st.markdown("---")
    
    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
