"""
Machine Learning Anomaly Detection for Industrial Sensor Data
============================================================

This script implements traditional ML models for anomaly detection including:
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- Elliptic Envelope

Part 2: Advanced ML Models for Anomaly Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MLAnomalyDetector:
    def __init__(self, data_path):
        """Initialize the ML anomaly detector."""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.X_scaled = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
        # Core sensor features
        self.core_features = ['temperature', 'vibration', 'pressure', 'flow_rate', 
                             'power_consumption', 'oil_level', 'bearing_temperature']
        
        # Additional engineered features
        self.engineered_features = []
        
    def load_and_prepare_data(self):
        """Load data and prepare features for ML models."""
        print("Loading and preparing data for ML models...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data.set_index('timestamp', inplace=True)
        
        print(f"Data loaded: {self.data.shape[0]:,} records")
        
        # Prepare feature matrix
        self.X = self.data[self.core_features].copy()
        
        # Add engineered features
        self._create_engineered_features()
        
        # Combine all features
        all_features = self.core_features + self.engineered_features
        self.X = self.data[all_features].copy()
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Features used: {list(self.X.columns)}")
        
        # Handle any missing values
        if self.X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.X = self.X.fillna(self.X.median())
        
        return self.X
    
    def _create_engineered_features(self):
        """Create additional engineered features for better anomaly detection."""
        print("Creating engineered features...")
        
        # Rolling statistics (already present in data)
        rolling_features = [col for col in self.data.columns if 'rolling_mean' in col or 'rolling_std' in col]
        self.engineered_features.extend(rolling_features)
        
        # Rate of change features (already present in data)
        rate_features = [col for col in self.data.columns if 'rate_of_change' in col]
        self.engineered_features.extend(rate_features)
        
        # Z-score features (already present in data)
        zscore_features = [col for col in self.data.columns if 'z_score' in col]
        self.engineered_features.extend(zscore_features)
        
        # Ratio features (already present in data)
        ratio_features = [col for col in self.data.columns if 'ratio' in col or 'product' in col]
        self.engineered_features.extend(ratio_features)
        
        # Time-based features
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['day_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['day_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        self.engineered_features.extend(time_features)
        
        # Interaction features
        self.data['temp_vibration_interaction'] = self.data['temperature'] * self.data['vibration']
        self.data['pressure_flow_interaction'] = self.data['pressure'] * self.data['flow_rate']
        self.data['power_oil_interaction'] = self.data['power_consumption'] * self.data['oil_level']
        
        interaction_features = ['temp_vibration_interaction', 'pressure_flow_interaction', 'power_oil_interaction']
        self.engineered_features.extend(interaction_features)
        
        print(f"Created {len(self.engineered_features)} engineered features")
    
    def scale_features(self, method='standard'):
        """Scale features for ML models."""
        print(f"Scaling features using {method} method...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"Features scaled. Shape: {self.X_scaled.shape}")
        return self.X_scaled
    
    def train_isolation_forest(self, contamination=0.1, random_state=42):
        """Train Isolation Forest model."""
        print("\n" + "="*50)
        print("TRAINING ISOLATION FOREST MODEL")
        print("="*50)
        
        # Create model
        model = IsolationForest(
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
        model.fit(self.X_scaled)
        
        # Make predictions
        predictions = model.predict(self.X_scaled)
        anomaly_scores = model.decision_function(self.X_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal)
        predictions_binary = np.where(predictions == -1, 1, 0)  # 1 for anomaly, 0 for normal
        
        self.models['isolation_forest'] = model
        self.results['isolation_forest'] = {
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
        
        return model, predictions_binary, anomaly_scores
    
    def train_one_class_svm(self, nu=0.1, kernel='rbf', gamma='scale'):
        """Train One-Class SVM model."""
        print("\n" + "="*50)
        print("TRAINING ONE-CLASS SVM MODEL")
        print("="*50)
        
        # Create model
        model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        
        # Train model
        print("Training One-Class SVM...")
        model.fit(self.X_scaled)
        
        # Make predictions
        predictions = model.predict(self.X_scaled)
        anomaly_scores = model.decision_function(self.X_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal)
        predictions_binary = np.where(predictions == -1, 1, 0)  # 1 for anomaly, 0 for normal
        
        self.models['one_class_svm'] = model
        self.results['one_class_svm'] = {
            'predictions': predictions_binary,
            'scores': anomaly_scores,
            'nu': nu
        }
        
        # Calculate metrics
        n_anomalies = np.sum(predictions_binary)
        anomaly_rate = n_anomalies / len(predictions_binary) * 100
        
        print(f"One-Class SVM Results:")
        print(f"  Anomalies detected: {n_anomalies:,}")
        print(f"  Anomaly rate: {anomaly_rate:.2f}%")
        print(f"  Expected contamination: {nu*100:.1f}%")
        
        return model, predictions_binary, anomaly_scores
    
    def train_local_outlier_factor(self, contamination=0.1, n_neighbors=20):
        """Train Local Outlier Factor model."""
        print("\n" + "="*50)
        print("TRAINING LOCAL OUTLIER FACTOR MODEL")
        print("="*50)
        
        # Create model
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False  # We're fitting on the same data we're predicting
        )
        
        # Make predictions (LOF doesn't have a separate fit method)
        print("Computing Local Outlier Factor...")
        predictions = model.fit_predict(self.X_scaled)
        anomaly_scores = model.negative_outlier_factor_
        
        # Convert predictions (-1 for anomaly, 1 for normal)
        predictions_binary = np.where(predictions == -1, 1, 0)  # 1 for anomaly, 0 for normal
        
        self.models['local_outlier_factor'] = model
        self.results['local_outlier_factor'] = {
            'predictions': predictions_binary,
            'scores': anomaly_scores,
            'contamination': contamination
        }
        
        # Calculate metrics
        n_anomalies = np.sum(predictions_binary)
        anomaly_rate = n_anomalies / len(predictions_binary) * 100
        
        print(f"Local Outlier Factor Results:")
        print(f"  Anomalies detected: {n_anomalies:,}")
        print(f"  Anomaly rate: {anomaly_rate:.2f}%")
        print(f"  Expected contamination: {contamination*100:.1f}%")
        
        return model, predictions_binary, anomaly_scores
    
    def train_elliptic_envelope(self, contamination=0.1, random_state=42):
        """Train Elliptic Envelope model."""
        print("\n" + "="*50)
        print("TRAINING ELLIPTIC ENVELOPE MODEL")
        print("="*50)
        
        # Create model
        model = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state,
            store_precision=True,
            assume_centered=False
        )
        
        # Train model
        print("Training Elliptic Envelope...")
        model.fit(self.X_scaled)
        
        # Make predictions
        predictions = model.predict(self.X_scaled)
        anomaly_scores = model.decision_function(self.X_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal)
        predictions_binary = np.where(predictions == -1, 1, 0)  # 1 for anomaly, 0 for normal
        
        self.models['elliptic_envelope'] = model
        self.results['elliptic_envelope'] = {
            'predictions': predictions_binary,
            'scores': anomaly_scores,
            'contamination': contamination
        }
        
        # Calculate metrics
        n_anomalies = np.sum(predictions_binary)
        anomaly_rate = n_anomalies / len(predictions_binary) * 100
        
        print(f"Elliptic Envelope Results:")
        print(f"  Anomalies detected: {n_anomalies:,}")
        print(f"  Anomaly rate: {anomaly_rate:.2f}%")
        print(f"  Expected contamination: {contamination*100:.1f}%")
        
        return model, predictions_binary, anomaly_scores
    
    def train_all_models(self, contamination=0.1):
        """Train all ML models."""
        print("\n" + "="*60)
        print("TRAINING ALL ML MODELS")
        print("="*60)
        
        # Train each model
        self.train_isolation_forest(contamination=contamination)
        self.train_one_class_svm(nu=contamination)
        self.train_local_outlier_factor(contamination=contamination)
        self.train_elliptic_envelope(contamination=contamination)
        
        print("\nAll models trained successfully!")
        return self.models, self.results
    
    def compare_models(self):
        """Compare performance of all trained models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            predictions = results['predictions']
            n_anomalies = np.sum(predictions)
            anomaly_rate = n_anomalies / len(predictions) * 100
            
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Anomalies': n_anomalies,
                'Anomaly Rate (%)': anomaly_rate,
                'Expected Rate (%)': results.get('contamination', results.get('nu', 0.1)) * 100
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def visualize_results(self, sample_size=1000):
        """Visualize anomaly detection results."""
        print("\n" + "="*50)
        print("VISUALIZING RESULTS")
        print("="*50)
        
        # Sample data for visualization
        if len(self.data) > sample_size:
            sample_indices = np.random.choice(len(self.data), sample_size, replace=False)
            sample_data = self.data.iloc[sample_indices]
            sample_X = self.X.iloc[sample_indices]
        else:
            sample_data = self.data
            sample_X = self.X
        
        # Create subplots for each model
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        model_names = list(self.results.keys())
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if i >= 4:  # Only show first 4 models
                break
                
            predictions = results['predictions']
            scores = results['scores']
            
            # Sample predictions and scores
            if len(self.data) > sample_size:
                sample_predictions = predictions[sample_indices]
                sample_scores = scores[sample_indices]
            else:
                sample_predictions = predictions
                sample_scores = scores
            
            # Plot anomalies
            normal_mask = sample_predictions == 0
            anomaly_mask = sample_predictions == 1
            
            # Use first two features for 2D visualization
            feature1, feature2 = self.core_features[0], self.core_features[1]
            
            axes[i].scatter(sample_X[feature1][normal_mask], 
                          sample_X[feature2][normal_mask], 
                          c='blue', alpha=0.6, s=20, label='Normal')
            
            axes[i].scatter(sample_X[feature1][anomaly_mask], 
                          sample_X[feature2][anomaly_mask], 
                          c='red', alpha=0.8, s=50, label='Anomaly')
            
            axes[i].set_xlabel(feature1.replace('_', ' ').title())
            axes[i].set_ylabel(feature2.replace('_', ' ').title())
            axes[i].set_title(f'{model_name.replace("_", " ").title()} - Anomaly Detection')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot anomaly scores distribution
        plt.figure(figsize=(15, 10))
        for i, (model_name, results) in enumerate(self.results.items()):
            plt.subplot(2, 2, i+1)
            scores = results['scores']
            predictions = results['predictions']
            
            # Plot score distribution
            plt.hist(scores[predictions == 0], bins=50, alpha=0.7, label='Normal', color='blue')
            plt.hist(scores[predictions == 1], bins=50, alpha=0.7, label='Anomaly', color='red')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.title(f'{model_name.replace("_", " ").title()} - Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_anomaly_patterns(self):
        """Analyze patterns in detected anomalies."""
        print("\n" + "="*50)
        print("ANOMALY PATTERN ANALYSIS")
        print("="*50)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name.replace('_', ' ').title()} Analysis:")
            
            predictions = results['predictions']
            anomaly_indices = np.where(predictions == 1)[0]
            
            if len(anomaly_indices) > 0:
                anomaly_data = self.data.iloc[anomaly_indices]
                
                # Time-based analysis
                print(f"  Time distribution of anomalies:")
                hourly_counts = anomaly_data.groupby(anomaly_data.index.hour).size()
                print(f"    Peak hour: {hourly_counts.idxmax()} ({hourly_counts.max()} anomalies)")
                
                # Sensor value analysis
                print(f"  Sensor values in anomalies:")
                for sensor in self.core_features:
                    sensor_values = anomaly_data[sensor]
                    print(f"    {sensor}: {sensor_values.min():.2f} - {sensor_values.max():.2f}")
    
    def generate_ml_report(self):
        """Generate comprehensive ML model report."""
        print("\n" + "="*60)
        print("MACHINE LEARNING ANOMALY DETECTION REPORT")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"  Total records: {len(self.data):,}")
        print(f"  Features used: {self.X.shape[1]}")
        print(f"  Core sensors: {len(self.core_features)}")
        print(f"  Engineered features: {len(self.engineered_features)}")
        
        print(f"\nModel Performance Summary:")
        comparison_df = self.compare_models()
        
        print(f"\nFeature Importance (Isolation Forest):")
        if 'isolation_forest' in self.models:
            model = self.models['isolation_forest']
            feature_importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
            
            feature_importance_df = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            print(feature_importance_df.head(10).to_string(index=False))
        
        print(f"\nRecommendations:")
        print(f"  1. Isolation Forest shows good performance for multivariate anomaly detection")
        print(f"  2. Consider ensemble methods combining multiple models")
        print(f"  3. Implement real-time scoring for new data points")
        print(f"  4. Fine-tune contamination parameters based on business requirements")
        print(f"  5. Monitor model performance over time and retrain as needed")

def main():
    """Main execution function."""
    print("Machine Learning Anomaly Detection for Industrial Sensor Data")
    print("=" * 70)
    
    # Initialize ML detector
    detector = MLAnomalyDetector('industrial_sensor_data_sample_clean.csv')
    
    # Load and prepare data
    X = detector.load_and_prepare_data()
    
    # Scale features
    detector.scale_features(method='standard')
    
    # Train all models
    models, results = detector.train_all_models(contamination=0.1)
    
    # Compare models
    comparison_df = detector.compare_models()
    
    # Visualize results
    detector.visualize_results()
    
    # Analyze patterns
    detector.analyze_anomaly_patterns()
    
    # Generate report
    detector.generate_ml_report()
    
    print("\n" + "="*70)
    print("ML ANOMALY DETECTION COMPLETE!")
    print("="*70)
    print("Next steps:")
    print("1. Fine-tune model parameters based on business requirements")
    print("2. Implement model persistence for production deployment")
    print("3. Set up automated retraining pipeline")
    print("4. Create real-time anomaly detection API")

if __name__ == "__main__":
    main()
