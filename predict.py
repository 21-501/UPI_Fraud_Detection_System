import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Load training data
df = pd.read_csv("C:\\Users\\Marthala Padmaja\\OneDrive\\Desktop\\myapp\\Fraud_Detection_App\\fraud.csv")

# Drop non-numeric/categorical columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Define features and target
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Convert 'type' to numeric using one-hot encoding
X = pd.get_dummies(X, columns=['type'], drop_first=True)

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y_resampled)

# Save model and scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")
