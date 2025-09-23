"""
Isolation Forest Anomaly Detection for Industrial Sensor Data
============================================================

This script implements Isolation Forest for anomaly detection on industrial sensor data.
Focuses on practical implementation without complex visualizations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class IsolationForestDetector:
    def __init__(self, data_path):
        """Initialize the Isolation Forest detector."""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.X_scaled = None
        self.scaler = None
        self.model = None
        self.results = {}
        
        # Core sensor features
        self.core_features = ['temperature', 'vibration', 'pressure', 'flow_rate', 
                             'power_consumption', 'oil_level', 'bearing_temperature']
        
    def load_and_prepare_data(self):
        """Load data and prepare features for Isolation Forest."""
        print("Loading and preparing data for Isolation Forest...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data.set_index('timestamp', inplace=True)
        
        print(f"Data loaded: {self.data.shape[0]:,} records")
        
        # Prepare feature matrix with core sensors
        self.X = self.data[self.core_features].copy()
        
        # Add some engineered features that are already in the data
        engineered_features = []
        
        # Add rolling statistics
        rolling_features = [col for col in self.data.columns if 'rolling_mean' in col or 'rolling_std' in col]
        engineered_features.extend(rolling_features)
        
        # Add rate of change features
        rate_features = [col for col in self.data.columns if 'rate_of_change' in col]
        engineered_features.extend(rate_features)
        
        # Add z-score features
        zscore_features = [col for col in self.data.columns if 'z_score' in col]
        engineered_features.extend(zscore_features)
        
        # Add ratio features
        ratio_features = [col for col in self.data.columns if 'ratio' in col or 'product' in col]
        engineered_features.extend(ratio_features)
        
        # Combine all features
        all_features = self.core_features + engineered_features
        self.X = self.data[all_features].copy()
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Core features: {len(self.core_features)}")
        print(f"Engineered features: {len(engineered_features)}")
        
        # Handle any missing values
        if self.X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.X = self.X.fillna(self.X.median())
        
        return self.X
    
    def scale_features(self):
        """Scale features for Isolation Forest."""
        print("Scaling features...")
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"Features scaled. Shape: {self.X_scaled.shape}")
        return self.X_scaled
    
    def train_isolation_forest(self, contamination=0.1, random_state=42):
        """Train Isolation Forest model."""
        print("\n" + "="*60)
        print("TRAINING ISOLATION FOREST MODEL")
        print("="*60)
        
        # Create model
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1
        )
        
        # Train model
        print("Training Isolation Forest...")
        self.model.fit(self.X_scaled)
        
        # Make predictions
        predictions = self.model.predict(self.X_scaled)
        anomaly_scores = self.model.decision_function(self.X_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal)
        predictions_binary = np.where(predictions == -1, 1, 0)  # 1 for anomaly, 0 for normal
        
        self.results = {
            'predictions': predictions_binary,
            'scores': anomaly_scores,
            'contamination': contamination
        }
        
        # Calculate metrics
        n_anomalies = np.sum(predictions_binary)
        anomaly_rate = n_anomalies / len(predictions_binary) * 100
        
        print(f"Isolation Forest Results:")
        print(f"  Anomalies detected: {n_anomalies:,}")
        print(f"  Anomaly rate: {anomaly_rate:.2f}%")
        print(f"  Expected contamination: {contamination*100:.1f}%")
        
        return self.model, predictions_binary, anomaly_scores
    
    def analyze_feature_importance(self):
        """Analyze feature importance from Isolation Forest."""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Calculate feature importance
        feature_importance = np.mean([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("Top 15 Most Important Features:")
        print(importance_df.head(15).to_string(index=False))
        
        return importance_df
    
    def analyze_anomaly_patterns(self):
        """Analyze patterns in detected anomalies."""
        print("\n" + "="*50)
        print("ANOMALY PATTERN ANALYSIS")
        print("="*50)
        
        if not self.results:
            print("No results available!")
            return
        
        predictions = self.results['predictions']
        anomaly_indices = np.where(predictions == 1)[0]
        
        if len(anomaly_indices) == 0:
            print("No anomalies detected!")
            return
        
        print(f"Total anomalies detected: {len(anomaly_indices):,}")
        
        # Get anomaly data
        anomaly_data = self.data.iloc[anomaly_indices]
        
        # Time-based analysis
        print(f"\nTime-based Analysis:")
        hourly_counts = anomaly_data.groupby(anomaly_data.index.hour).size()
        daily_counts = anomaly_data.groupby(anomaly_data.index.date).size()
        
        print(f"  Peak anomaly hour: {hourly_counts.idxmax()} ({hourly_counts.max()} anomalies)")
        print(f"  Peak anomaly day: {daily_counts.idxmax()} ({daily_counts.max()} anomalies)")
        
        # Sensor value analysis for anomalies
        print(f"\nSensor Values in Anomalies:")
        for sensor in self.core_features:
            sensor_values = anomaly_data[sensor]
            normal_values = self.data[sensor]
            
            print(f"  {sensor.replace('_', ' ').title()}:")
            print(f"    Anomaly range: {sensor_values.min():.2f} - {sensor_values.max():.2f}")
            print(f"    Normal range: {normal_values.min():.2f} - {normal_values.max():.2f}")
            print(f"    Anomaly mean: {sensor_values.mean():.2f}")
            print(f"    Normal mean: {normal_values.mean():.2f}")
        
        # Multi-sensor anomaly analysis
        print(f"\nMulti-sensor Anomaly Analysis:")
        multi_anomaly_counts = {}
        
        for idx in anomaly_indices:
            row = self.data.iloc[idx]
            anomaly_count = 0
            
            for sensor in self.core_features:
                # Check if this sensor value is extreme
                sensor_values = self.data[sensor]
                q1 = sensor_values.quantile(0.25)
                q3 = sensor_values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                if row[sensor] < lower_bound or row[sensor] > upper_bound:
                    anomaly_count += 1
            
            multi_anomaly_counts[anomaly_count] = multi_anomaly_counts.get(anomaly_count, 0) + 1
        
        for sensor_count, record_count in sorted(multi_anomaly_counts.items()):
            print(f"  Records with {sensor_count} sensor anomalies: {record_count:,}")
    
    def compare_with_baseline(self):
        """Compare Isolation Forest results with baseline statistical methods."""
        print("\n" + "="*50)
        print("COMPARISON WITH BASELINE METHODS")
        print("="*50)
        
        if not self.results:
            print("No Isolation Forest results available!")
            return
        
        ml_predictions = self.results['predictions']
        ml_anomalies = np.sum(ml_predictions)
        
        print(f"Isolation Forest: {ml_anomalies:,} anomalies ({ml_anomalies/len(ml_predictions)*100:.2f}%)")
        
        # Compare with IQR method for each sensor
        print(f"\nComparison with IQR Method:")
        for sensor in self.core_features:
            sensor_values = self.data[sensor]
            q1 = sensor_values.quantile(0.25)
            q3 = sensor_values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            iqr_anomalies = ((sensor_values < lower_bound) | (sensor_values > upper_bound)).sum()
            
            print(f"  {sensor.replace('_', ' ').title()}:")
            print(f"    IQR method: {iqr_anomalies:,} anomalies ({iqr_anomalies/len(sensor_values)*100:.2f}%)")
    
    def generate_detailed_report(self):
        """Generate detailed anomaly detection report."""
        print("\n" + "="*60)
        print("ISOLATION FOREST ANOMALY DETECTION REPORT")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"  Total records: {len(self.data):,}")
        print(f"  Time period: {self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}")
        print(f"  Duration: {(self.data.index.max() - self.data.index.min()).days} days")
        print(f"  Features used: {self.X.shape[1]}")
        
        print(f"\nModel Configuration:")
        print(f"  Algorithm: Isolation Forest")
        print(f"  Contamination: {self.results.get('contamination', 'N/A')}")
        print(f"  N estimators: 100")
        print(f"  Max samples: auto")
        print(f"  Max features: 1.0")
        
        print(f"\nDetection Results:")
        if self.results:
            predictions = self.results['predictions']
            n_anomalies = np.sum(predictions)
            anomaly_rate = n_anomalies / len(predictions) * 100
            
            print(f"  Anomalies detected: {n_anomalies:,}")
            print(f"  Anomaly rate: {anomaly_rate:.2f}%")
            print(f"  Normal records: {len(predictions) - n_anomalies:,}")
        
        print(f"\nKey Insights:")
        print(f"  1. Isolation Forest provides multivariate anomaly detection")
        print(f"  2. Model considers interactions between all sensor features")
        print(f"  3. More sophisticated than univariate statistical methods")
        print(f"  4. Suitable for complex industrial sensor data")
        
        print(f"\nRecommendations:")
        print(f"  1. Use Isolation Forest for real-time anomaly detection")
        print(f"  2. Retrain model periodically with new data")
        print(f"  3. Combine with domain knowledge for validation")
        print(f"  4. Implement alerting system for detected anomalies")
        print(f"  5. Monitor model performance over time")
    
    def save_results(self, filename='isolation_forest_results.csv'):
        """Save anomaly detection results to CSV."""
        if not self.results:
            print("No results to save!")
            return
        
        # Create results dataframe
        results_df = self.data.copy()
        results_df['isolation_forest_anomaly'] = self.results['predictions']
        results_df['anomaly_score'] = self.results['scores']
        
        # Save to CSV
        results_df.to_csv(filename)
        print(f"Results saved to {filename}")
        
        return results_df

def main():
    """Main execution function."""
    print("Isolation Forest Anomaly Detection for Industrial Sensor Data")
    print("=" * 70)
    
    # Initialize detector
    detector = IsolationForestDetector('industrial_sensor_data_sample_clean.csv')
    
    # Load and prepare data
    X = detector.load_and_prepare_data()
    
    # Scale features
    detector.scale_features()
    
    # Train Isolation Forest
    model, predictions, scores = detector.train_isolation_forest(contamination=0.1)
    
    # Analyze feature importance
    importance_df = detector.analyze_feature_importance()
    
    # Analyze anomaly patterns
    detector.analyze_anomaly_patterns()
    
    # Compare with baseline
    detector.compare_with_baseline()
    
    # Generate detailed report
    detector.generate_detailed_report()
    
    # Save results
    results_df = detector.save_results()
    
    print("\n" + "="*70)
    print("ISOLATION FOREST TRAINING COMPLETE!")
    print("="*70)
    print("Next steps:")
    print("1. Review detected anomalies and validate with domain knowledge")
    print("2. Fine-tune contamination parameter based on business requirements")
    print("3. Implement real-time scoring for new data points")
    print("4. Set up automated retraining pipeline")
    print("5. Create production deployment with model persistence")

if __name__ == "__main__":
    main()
