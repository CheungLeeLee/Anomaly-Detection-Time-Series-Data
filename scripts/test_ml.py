"""
Test script for ML anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("Starting ML anomaly detection test...")

# Load data
print("Loading data...")
data = pd.read_csv('industrial_sensor_data_sample_clean.csv')
print(f"Data loaded: {data.shape}")

# Prepare features
core_features = ['temperature', 'vibration', 'pressure', 'flow_rate', 
                 'power_consumption', 'oil_level', 'bearing_temperature']

X = data[core_features].copy()
print(f"Feature matrix: {X.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Features scaled: {X_scaled.shape}")

# Train Isolation Forest
print("Training Isolation Forest...")
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_scaled)

# Make predictions
predictions = model.predict(X_scaled)
anomaly_scores = model.decision_function(X_scaled)

# Convert predictions
predictions_binary = np.where(predictions == -1, 1, 0)

# Results
n_anomalies = np.sum(predictions_binary)
anomaly_rate = n_anomalies / len(predictions_binary) * 100

print(f"\nResults:")
print(f"Anomalies detected: {n_anomalies:,}")
print(f"Anomaly rate: {anomaly_rate:.2f}%")

# Feature importance
feature_importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
importance_df = pd.DataFrame({
    'Feature': core_features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance:")
print(importance_df.to_string(index=False))

print("\nTest completed successfully!")
