import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("Starting minimal test...")

# Load data
print("Loading data...")
data_path = os.path.join(os.getcwd(), 'data', 'industrial_sensor_data_sample_clean.csv')
print(f"Data path: {data_path}")
print(f"File exists: {os.path.exists(data_path)}")

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
    print(f"Data loaded: {data.shape[0]:,} records, {data.shape[1]} columns")
    
    # Simple test
    core_features = ['temperature', 'vibration', 'pressure', 'flow_rate', 
                     'power_consumption', 'oil_level', 'bearing_temperature']
    
    X = data[core_features].copy()
    print(f"Feature matrix: {X.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Features scaled: {X_scaled.shape}")
    
    # Train simple model
    print("Training Isolation Forest...")
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_scaled)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    predictions_binary = np.where(predictions == -1, 1, 0)
    
    n_anomalies = np.sum(predictions_binary)
    print(f"Anomalies detected: {n_anomalies:,}")
    print(f"Anomaly rate: {n_anomalies/len(predictions_binary)*100:.2f}%")
    
    print("Test completed successfully!")
else:
    print("Data file not found!")
