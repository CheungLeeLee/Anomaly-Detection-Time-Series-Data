"""
Industrial Sensor Data Anomaly Detection Analysis
================================================

This script performs comprehensive analysis of industrial sensor data to detect anomalies
and build a baseline predictive model for device health monitoring.

Part 1: Data Analysis & Baseline Model
- Load and explore the time-series data
- Visualize data to identify obvious anomalies
- Build baseline anomaly detection model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SensorDataAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with sensor data."""
        self.data_path = data_path
        self.data = None
        self.anomalies = None
        
    def load_data(self):
        """Load the sensor data from CSV file."""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Set timestamp as index for time series analysis
        self.data.set_index('timestamp', inplace=True)
        
        print(f"Data loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Time span: {(self.data.index.max() - self.data.index.min()).days} days")
        
        return self.data
    
    def explore_data(self):
        """Perform initial data exploration."""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print("\nDataset Info:")
        print(f"Total records: {len(self.data):,}")
        print(f"Number of machines: {self.data['machine_id'].nunique()}")
        print(f"Unique machines: {self.data['machine_id'].unique()}")
        
        # Check for missing values
        print("\nMissing Values:")
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            print(missing_data[missing_data > 0])
        else:
            print("No missing values found!")
        
        # Basic statistics for sensor readings
        sensor_columns = ['temperature', 'vibration', 'pressure', 'flow_rate', 
                         'power_consumption', 'oil_level', 'bearing_temperature']
        
        print("\nSensor Data Statistics:")
        print(self.data[sensor_columns].describe())
        
        # Check maintenance status distribution
        print("\nMaintenance Status Distribution:")
        print(self.data['maintenance_status'].value_counts())
        
        return sensor_columns
    
    def visualize_time_series(self, sensor_columns, sample_size=1000):
        """Create comprehensive time series visualizations."""
        print("\n" + "="*50)
        print("TIME SERIES VISUALIZATION")
        print("="*50)
        
        # Sample data for visualization if dataset is large
        if len(self.data) > sample_size:
            plot_data = self.data.sample(n=sample_size).sort_index()
            print(f"Using sample of {sample_size} records for visualization")
        else:
            plot_data = self.data
        
        # Create subplots for each sensor
        fig, axes = plt.subplots(len(sensor_columns), 1, figsize=(15, 3*len(sensor_columns)))
        if len(sensor_columns) == 1:
            axes = [axes]
        
        for i, sensor in enumerate(sensor_columns):
            axes[i].plot(plot_data.index, plot_data[sensor], alpha=0.7, linewidth=0.8)
            axes[i].set_title(f'{sensor.replace("_", " ").title()} Over Time', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(sensor.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(range(len(plot_data)), plot_data[sensor], 1)
            p = np.poly1d(z)
            axes[i].plot(plot_data.index, p(range(len(plot_data))), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.show()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data[sensor_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Sensor Data Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return plot_data
    
    def detect_statistical_anomalies(self, sensor_columns, method='iqr', threshold=1.5):
        """Detect anomalies using statistical methods."""
        print("\n" + "="*50)
        print("STATISTICAL ANOMALY DETECTION")
        print("="*50)
        
        anomalies = {}
        
        for sensor in sensor_columns:
            if method == 'iqr':
                # Interquartile Range method
                Q1 = self.data[sensor].quantile(0.25)
                Q3 = self.data[sensor].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                sensor_anomalies = self.data[(self.data[sensor] < lower_bound) | 
                                           (self.data[sensor] > upper_bound)]
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((self.data[sensor] - self.data[sensor].mean()) / self.data[sensor].std())
                sensor_anomalies = self.data[z_scores > threshold]
            
            anomalies[sensor] = sensor_anomalies
            
            print(f"\n{sensor.replace('_', ' ').title()} Anomalies ({method.upper()} method):")
            print(f"  Total anomalies: {len(sensor_anomalies):,}")
            print(f"  Percentage: {len(sensor_anomalies)/len(self.data)*100:.2f}%")
            if len(sensor_anomalies) > 0:
                print(f"  Min value: {sensor_anomalies[sensor].min():.2f}")
                print(f"  Max value: {sensor_anomalies[sensor].max():.2f}")
        
        self.anomalies = anomalies
        return anomalies
    
    def visualize_anomalies(self, sensor_columns, sample_size=2000):
        """Visualize detected anomalies."""
        print("\n" + "="*50)
        print("ANOMALY VISUALIZATION")
        print("="*50)
        
        # Sample data for visualization
        if len(self.data) > sample_size:
            plot_data = self.data.sample(n=sample_size).sort_index()
        else:
            plot_data = self.data
        
        # Create subplots for anomaly visualization
        fig, axes = plt.subplots(len(sensor_columns), 1, figsize=(15, 3*len(sensor_columns)))
        if len(sensor_columns) == 1:
            axes = [axes]
        
        for i, sensor in enumerate(sensor_columns):
            # Plot normal data
            normal_data = plot_data[~plot_data.index.isin(self.anomalies[sensor].index)]
            axes[i].scatter(normal_data.index, normal_data[sensor], 
                           alpha=0.6, s=20, color='blue', label='Normal')
            
            # Plot anomalies
            sensor_anomalies_sample = self.anomalies[sensor][self.anomalies[sensor].index.isin(plot_data.index)]
            if len(sensor_anomalies_sample) > 0:
                axes[i].scatter(sensor_anomalies_sample.index, sensor_anomalies_sample[sensor], 
                               alpha=0.8, s=50, color='red', label='Anomaly')
            
            axes[i].set_title(f'{sensor.replace("_", " ").title()} - Anomaly Detection', 
                            fontsize=12, fontweight='bold')
            axes[i].set_ylabel(sensor.replace("_", " ").title())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Create box plots to show anomaly distribution
        plt.figure(figsize=(15, 8))
        for i, sensor in enumerate(sensor_columns):
            plt.subplot(2, 4, i+1)
            plt.boxplot(self.data[sensor], vert=True)
            plt.title(f'{sensor.replace("_", " ").title()}')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def build_baseline_model(self, sensor_columns):
        """Build a baseline anomaly detection model."""
        print("\n" + "="*50)
        print("BASELINE MODEL CONSTRUCTION")
        print("="*50)
        
        # Create a simple baseline model using multiple statistical methods
        baseline_results = {}
        
        for sensor in sensor_columns:
            # Method 1: Z-score based
            z_scores = np.abs((self.data[sensor] - self.data[sensor].mean()) / self.data[sensor].std())
            z_anomalies = z_scores > 2.5
            
            # Method 2: IQR based
            Q1 = self.data[sensor].quantile(0.25)
            Q3 = self.data[sensor].quantile(0.75)
            IQR = Q3 - Q1
            iqr_anomalies = (self.data[sensor] < Q1 - 1.5*IQR) | (self.data[sensor] > Q3 + 1.5*IQR)
            
            # Method 3: Percentile based
            p5, p95 = self.data[sensor].quantile([0.05, 0.95])
            percentile_anomalies = (self.data[sensor] < p5) | (self.data[sensor] > p95)
            
            # Combine methods (OR logic - if any method flags as anomaly)
            combined_anomalies = z_anomalies | iqr_anomalies | percentile_anomalies
            
            baseline_results[sensor] = {
                'z_score_anomalies': z_anomalies.sum(),
                'iqr_anomalies': iqr_anomalies.sum(),
                'percentile_anomalies': percentile_anomalies.sum(),
                'combined_anomalies': combined_anomalies.sum(),
                'anomaly_rate': combined_anomalies.mean() * 100
            }
            
            print(f"\n{sensor.replace('_', ' ').title()} Baseline Model Results:")
            print(f"  Z-score anomalies: {z_anomalies.sum():,}")
            print(f"  IQR anomalies: {iqr_anomalies.sum():,}")
            print(f"  Percentile anomalies: {percentile_anomalies.sum():,}")
            print(f"  Combined anomalies: {combined_anomalies.sum():,}")
            print(f"  Anomaly rate: {combined_anomalies.mean()*100:.2f}%")
        
        return baseline_results
    
    def generate_summary_report(self, sensor_columns):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("ANOMALY DETECTION SUMMARY REPORT")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  Total records: {len(self.data):,}")
        print(f"  Time period: {self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}")
        print(f"  Duration: {(self.data.index.max() - self.data.index.min()).days} days")
        print(f"  Machines monitored: {self.data['machine_id'].nunique()}")
        
        print(f"\nSensor Monitoring Summary:")
        for sensor in sensor_columns:
            print(f"  {sensor.replace('_', ' ').title()}:")
            print(f"    Range: {self.data[sensor].min():.2f} - {self.data[sensor].max():.2f}")
            print(f"    Mean: {self.data[sensor].mean():.2f}")
            print(f"    Std: {self.data[sensor].std():.2f}")
            print(f"    CV: {(self.data[sensor].std() / self.data[sensor].mean() * 100):.2f}%")
        
        print(f"\nAnomaly Detection Results:")
        total_anomalies = 0
        for sensor in sensor_columns:
            if self.anomalies and sensor in self.anomalies:
                anomaly_count = len(self.anomalies[sensor])
                total_anomalies += anomaly_count
                print(f"  {sensor.replace('_', ' ').title()}: {anomaly_count:,} anomalies ({anomaly_count/len(self.data)*100:.2f}%)")
        
        print(f"\nOverall Anomaly Rate: {total_anomalies/len(self.data)/len(sensor_columns)*100:.2f}%")
        
        print(f"\nRecommendations:")
        print(f"  1. Focus on sensors with highest anomaly rates for maintenance")
        print(f"  2. Implement real-time monitoring for critical sensors")
        print(f"  3. Set up automated alerts for anomaly detection")
        print(f"  4. Consider machine learning models for improved accuracy")

def main():
    """Main execution function."""
    print("Industrial Sensor Data Anomaly Detection Analysis")
    print("=" * 60)
    
    # Initialize analyzer with sample data
    analyzer = SensorDataAnalyzer('industrial_sensor_data_sample_clean.csv')
    
    # Load and explore data
    data = analyzer.load_data()
    sensor_columns = analyzer.explore_data()
    
    # Visualize time series data
    analyzer.visualize_time_series(sensor_columns)
    
    # Detect anomalies using statistical methods
    analyzer.detect_statistical_anomalies(sensor_columns, method='iqr')
    
    # Visualize anomalies
    analyzer.visualize_anomalies(sensor_columns)
    
    # Build baseline model
    baseline_results = analyzer.build_baseline_model(sensor_columns)
    
    # Generate summary report
    analyzer.generate_summary_report(sensor_columns)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Review anomaly patterns and identify root causes")
    print("2. Implement advanced ML models (Isolation Forest, One-Class SVM)")
    print("3. Set up real-time monitoring system")
    print("4. Create automated alerting system")

if __name__ == "__main__":
    main()
