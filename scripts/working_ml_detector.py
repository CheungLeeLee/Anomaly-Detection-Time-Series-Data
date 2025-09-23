"""
Working ML Anomaly Detection Script
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 60)
    print("ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 60)
    print("Script started successfully!")
    
    # Load data
    print("\n1. Loading data...")
    # Use current working directory (project root)
    data_path = os.path.join(os.getcwd(), 'data', 'industrial_sensor_data_sample_clean.csv')
    data = pd.read_csv(data_path)
    print(f"   Data loaded: {data.shape[0]:,} records, {data.shape[1]} columns")
    
    # Prepare features
    print("\n2. Preparing features...")
    core_features = ['temperature', 'vibration', 'pressure', 'flow_rate', 
                     'power_consumption', 'oil_level', 'bearing_temperature']
    
    X = data[core_features].copy()
    print(f"   Feature matrix: {X.shape}")
    print(f"   Features: {core_features}")
    
    # Check for missing values
    missing_values = X.isnull().sum().sum()
    if missing_values > 0:
        print(f"   Warning: {missing_values} missing values found")
        X = X.fillna(X.median())
        print("   Missing values filled with median")
    else:
        print("   No missing values found")
    
    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Features scaled: {X_scaled.shape}")
    
    # Train Isolation Forest
    print("\n4. Training Isolation Forest...")
    contamination = 0.1  # Expect 10% anomalies
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1
    )
    
    model.fit(X_scaled)
    print(f"   Model trained with contamination={contamination}")
    
    # Make predictions
    print("\n5. Making predictions...")
    predictions = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)
    
    # Convert predictions (-1 for anomaly, 1 for normal)
    predictions_binary = np.where(predictions == -1, 1, 0)  # 1 for anomaly, 0 for normal
    
    # Calculate results
    n_anomalies = np.sum(predictions_binary)
    anomaly_rate = n_anomalies / len(predictions_binary) * 100
    
    print(f"\n6. RESULTS:")
    print(f"   Anomalies detected: {n_anomalies:,}")
    print(f"   Anomaly rate: {anomaly_rate:.2f}%")
    print(f"   Expected rate: {contamination*100:.1f}%")
    print(f"   Normal records: {len(predictions_binary) - n_anomalies:,}")
    
    # Feature importance
    print(f"\n7. FEATURE IMPORTANCE:")
    feature_importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': core_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    
    # Analyze anomalies
    print(f"\n8. ANOMALY ANALYSIS:")
    anomaly_indices = np.where(predictions_binary == 1)[0]
    
    if len(anomaly_indices) > 0:
        anomaly_data = data.iloc[anomaly_indices]
        
        print(f"   Anomaly records: {len(anomaly_indices):,}")
        
        # Sensor values in anomalies
        print(f"\n   Sensor values in anomalies:")
        for sensor in core_features:
            sensor_values = anomaly_data[sensor]
            normal_values = data[sensor]
            
            print(f"     {sensor.replace('_', ' ').title()}:")
            print(f"       Anomaly range: {sensor_values.min():.2f} - {sensor_values.max():.2f}")
            print(f"       Normal range: {normal_values.min():.2f} - {normal_values.max():.2f}")
            print(f"       Anomaly mean: {sensor_values.mean():.2f}")
            print(f"       Normal mean: {normal_values.mean():.2f}")
    
    # Save results
    print(f"\n9. SAVING RESULTS...")
    results_df = data.copy()
    results_df['isolation_forest_anomaly'] = predictions_binary
    results_df['anomaly_score'] = anomaly_scores
    
    results_path = os.path.join(os.getcwd(), 'results', 'isolation_forest_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"   Results saved to 'isolation_forest_results.csv'")
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    print(f"\nKey Findings:")
    print(f"• Isolation Forest detected {n_anomalies:,} anomalies ({anomaly_rate:.2f}%)")
    print(f"• Most important features: {importance_df.iloc[0]['Feature']}, {importance_df.iloc[1]['Feature']}")
    print(f"• Model considers multivariate relationships between sensors")
    print(f"• Results saved for further analysis")
    
    print(f"\nNext Steps:")
    print(f"1. Review detected anomalies for business relevance")
    print(f"2. Fine-tune contamination parameter if needed")
    print(f"3. Implement real-time scoring for new data")
    print(f"4. Set up automated retraining pipeline")

if __name__ == "__main__":
    main()
