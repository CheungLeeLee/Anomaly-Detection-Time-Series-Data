"""
Simple Industrial Sensor Data Anomaly Detection Analysis
======================================================

This script performs basic analysis of industrial sensor data using only built-in Python libraries
and basic statistical methods to detect anomalies.

Part 1: Data Analysis & Baseline Model
- Load and explore the time-series data
- Identify obvious anomalies using statistical methods
- Build baseline anomaly detection model
"""

import csv
import statistics
import math
from collections import defaultdict
from datetime import datetime

class SimpleSensorAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with sensor data."""
        self.data_path = data_path
        self.data = []
        self.sensor_columns = ['temperature', 'vibration', 'pressure', 'flow_rate', 
                              'power_consumption', 'oil_level', 'bearing_temperature']
        
    def load_data(self):
        """Load the sensor data from CSV file."""
        print(f"Loading data from {self.data_path}...")
        
        with open(self.data_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert timestamp
                row['timestamp'] = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                
                # Convert numeric columns
                for col in self.sensor_columns:
                    row[col] = float(row[col])
                
                self.data.append(row)
        
        print(f"Data loaded successfully!")
        print(f"Total records: {len(self.data):,}")
        print(f"Date range: {self.data[0]['timestamp']} to {self.data[-1]['timestamp']}")
        
        return self.data
    
    def explore_data(self):
        """Perform initial data exploration."""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        machines = set(row['machine_id'] for row in self.data)
        print(f"Number of machines: {len(machines)}")
        print(f"Unique machines: {list(machines)}")
        
        # Check maintenance status distribution
        maintenance_counts = defaultdict(int)
        for row in self.data:
            maintenance_counts[row['maintenance_status']] += 1
        
        print("\nMaintenance Status Distribution:")
        for status, count in maintenance_counts.items():
            print(f"  {status}: {count:,} ({count/len(self.data)*100:.1f}%)")
        
        # Basic statistics for sensor readings
        print("\nSensor Data Statistics:")
        for sensor in self.sensor_columns:
            values = [row[sensor] for row in self.data]
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            min_val = min(values)
            max_val = max(values)
            
            print(f"\n{sensor.replace('_', ' ').title()}:")
            print(f"  Count: {len(values):,}")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Std: {std_val:.2f}")
            print(f"  Min: {min_val:.2f}")
            print(f"  Max: {max_val:.2f}")
            print(f"  Range: {max_val - min_val:.2f}")
            print(f"  CV: {(std_val/mean_val*100):.2f}%")
    
    def detect_statistical_anomalies(self, method='iqr', threshold=1.5):
        """Detect anomalies using statistical methods."""
        print("\n" + "="*50)
        print("STATISTICAL ANOMALY DETECTION")
        print("="*50)
        
        anomalies = {}
        
        for sensor in self.sensor_columns:
            values = [row[sensor] for row in self.data]
            
            if method == 'iqr':
                # Interquartile Range method
                sorted_values = sorted(values)
                n = len(sorted_values)
                q1_idx = int(n * 0.25)
                q3_idx = int(n * 0.75)
                
                Q1 = sorted_values[q1_idx]
                Q3 = sorted_values[q3_idx]
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                sensor_anomalies = []
                for i, row in enumerate(self.data):
                    if row[sensor] < lower_bound or row[sensor] > upper_bound:
                        sensor_anomalies.append(i)
                
            elif method == 'zscore':
                # Z-score method
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                sensor_anomalies = []
                for i, row in enumerate(self.data):
                    z_score = abs((row[sensor] - mean_val) / std_val) if std_val > 0 else 0
                    if z_score > threshold:
                        sensor_anomalies.append(i)
            
            anomalies[sensor] = sensor_anomalies
            
            print(f"\n{sensor.replace('_', ' ').title()} Anomalies ({method.upper()} method):")
            print(f"  Total anomalies: {len(sensor_anomalies):,}")
            print(f"  Percentage: {len(sensor_anomalies)/len(self.data)*100:.2f}%")
            
            if len(sensor_anomalies) > 0:
                anomaly_values = [self.data[i][sensor] for i in sensor_anomalies]
                print(f"  Min anomaly value: {min(anomaly_values):.2f}")
                print(f"  Max anomaly value: {max(anomaly_values):.2f}")
        
        return anomalies
    
    def analyze_anomaly_patterns(self, anomalies):
        """Analyze patterns in detected anomalies."""
        print("\n" + "="*50)
        print("ANOMALY PATTERN ANALYSIS")
        print("="*50)
        
        # Find records that are anomalies in multiple sensors
        multi_sensor_anomalies = defaultdict(int)
        
        for sensor, anomaly_indices in anomalies.items():
            for idx in anomaly_indices:
                multi_sensor_anomalies[idx] += 1
        
        # Count how many records are anomalies in multiple sensors
        multi_counts = defaultdict(int)
        for idx, count in multi_sensor_anomalies.items():
            multi_counts[count] += 1
        
        print("Multi-sensor anomaly distribution:")
        for sensor_count, record_count in sorted(multi_counts.items()):
            print(f"  Anomalies in {sensor_count} sensors: {record_count:,} records")
        
        # Find the most problematic records
        if multi_sensor_anomalies:
            max_anomalies = max(multi_sensor_anomalies.values())
            worst_records = [idx for idx, count in multi_sensor_anomalies.items() if count == max_anomalies]
            
            print(f"\nMost problematic records (anomalies in {max_anomalies} sensors):")
            for idx in worst_records[:5]:  # Show top 5
                row = self.data[idx]
                print(f"  {row['timestamp']}: {row['machine_id']} - {max_anomalies} sensor anomalies")
    
    def build_baseline_model(self):
        """Build a baseline anomaly detection model."""
        print("\n" + "="*50)
        print("BASELINE MODEL CONSTRUCTION")
        print("="*50)
        
        baseline_results = {}
        
        for sensor in self.sensor_columns:
            values = [row[sensor] for row in self.data]
            
            # Method 1: Z-score based (threshold = 2.5)
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            z_anomalies = 0
            for val in values:
                z_score = abs((val - mean_val) / std_val) if std_val > 0 else 0
                if z_score > 2.5:
                    z_anomalies += 1
            
            # Method 2: IQR based
            sorted_values = sorted(values)
            n = len(sorted_values)
            q1_idx = int(n * 0.25)
            q3_idx = int(n * 0.75)
            Q1 = sorted_values[q1_idx]
            Q3 = sorted_values[q3_idx]
            IQR = Q3 - Q1
            iqr_anomalies = 0
            for val in values:
                if val < Q1 - 1.5*IQR or val > Q3 + 1.5*IQR:
                    iqr_anomalies += 1
            
            # Method 3: Percentile based
            p5_idx = int(n * 0.05)
            p95_idx = int(n * 0.95)
            p5 = sorted_values[p5_idx]
            p95 = sorted_values[p95_idx]
            percentile_anomalies = 0
            for val in values:
                if val < p5 or val > p95:
                    percentile_anomalies += 1
            
            baseline_results[sensor] = {
                'z_score_anomalies': z_anomalies,
                'iqr_anomalies': iqr_anomalies,
                'percentile_anomalies': percentile_anomalies,
                'total_records': len(values)
            }
            
            print(f"\n{sensor.replace('_', ' ').title()} Baseline Model Results:")
            print(f"  Z-score anomalies: {z_anomalies:,} ({z_anomalies/len(values)*100:.2f}%)")
            print(f"  IQR anomalies: {iqr_anomalies:,} ({iqr_anomalies/len(values)*100:.2f}%)")
            print(f"  Percentile anomalies: {percentile_anomalies:,} ({percentile_anomalies/len(values)*100:.2f}%)")
        
        return baseline_results
    
    def generate_summary_report(self, anomalies, baseline_results):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("ANOMALY DETECTION SUMMARY REPORT")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  Total records: {len(self.data):,}")
        print(f"  Time period: {self.data[0]['timestamp'].strftime('%Y-%m-%d')} to {self.data[-1]['timestamp'].strftime('%Y-%m-%d')}")
        print(f"  Duration: {(self.data[-1]['timestamp'] - self.data[0]['timestamp']).days} days")
        
        machines = set(row['machine_id'] for row in self.data)
        print(f"  Machines monitored: {len(machines)}")
        
        print(f"\nSensor Monitoring Summary:")
        for sensor in self.sensor_columns:
            values = [row[sensor] for row in self.data]
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            print(f"  {sensor.replace('_', ' ').title()}:")
            print(f"    Range: {min(values):.2f} - {max(values):.2f}")
            print(f"    Mean: {mean_val:.2f}")
            print(f"    Std: {std_val:.2f}")
            print(f"    CV: {(std_val/mean_val*100):.2f}%")
        
        print(f"\nAnomaly Detection Results:")
        total_anomalies = 0
        for sensor in self.sensor_columns:
            if sensor in anomalies:
                anomaly_count = len(anomalies[sensor])
                total_anomalies += anomaly_count
                print(f"  {sensor.replace('_', ' ').title()}: {anomaly_count:,} anomalies ({anomaly_count/len(self.data)*100:.2f}%)")
        
        print(f"\nOverall Anomaly Rate: {total_anomalies/len(self.data)/len(self.sensor_columns)*100:.2f}%")
        
        print(f"\nRecommendations:")
        print(f"  1. Focus on sensors with highest anomaly rates for maintenance")
        print(f"  2. Implement real-time monitoring for critical sensors")
        print(f"  3. Set up automated alerts for anomaly detection")
        print(f"  4. Consider machine learning models for improved accuracy")
        print(f"  5. Investigate multi-sensor anomalies as priority cases")

def main():
    """Main execution function."""
    print("Industrial Sensor Data Anomaly Detection Analysis")
    print("=" * 60)
    
    # Initialize analyzer with sample data
    analyzer = SimpleSensorAnalyzer('industrial_sensor_data_sample_clean.csv')
    
    # Load and explore data
    data = analyzer.load_data()
    analyzer.explore_data()
    
    # Detect anomalies using statistical methods
    anomalies = analyzer.detect_statistical_anomalies(method='iqr')
    
    # Analyze anomaly patterns
    analyzer.analyze_anomaly_patterns(anomalies)
    
    # Build baseline model
    baseline_results = analyzer.build_baseline_model()
    
    # Generate summary report
    analyzer.generate_summary_report(anomalies, baseline_results)
    
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
