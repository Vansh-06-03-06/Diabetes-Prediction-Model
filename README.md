# Diabetes Prediction Project

This is a machine learning project that predicts whether a patient has diabetes based on diagnostic measurements. The model is built using a Support Vector Machine (SVM) and deployed as a web application using Streamlit.

## Features
- Interactive web application using Streamlit.
- Uses `scikit-learn` for machine learning (`SVC` and `StandardScaler`).
- Predicts diabetes using user input parameters: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.

## How to Run the Web App

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. A browser window will open automatically with the web app running at `http://localhost:8501`.

## Files in this Repository
- `Diabetes_Prediction_Model.ipynb`: The Jupyter Notebook used for data exploration, preprocessing, and model training.
- `app.py`: The Streamlit web application script.
- `train_and_save.py`: Script to train the model and export it to `.sav` files.
- `diabetes_model.sav`: The saved SVM model.
- `scaler.sav`: The saved StandardScaler object used to normalize input data.
- `requirements.txt`: List of dependencies needed to run the project.
