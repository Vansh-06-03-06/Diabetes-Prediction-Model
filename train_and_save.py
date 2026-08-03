import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

# Load dataset
diabetes_dataset = pd.read_csv('diabetes - diabetes.csv')

# Separate features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Save model and scaler
pickle.dump(classifier, open('diabetes_model.sav', 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))

print("Model and Scaler successfully saved!")
