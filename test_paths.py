import os
import pandas as pd

# Test path resolution
print("Current directory:", os.getcwd())
print("Script file:", __file__)

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Project root:", project_root)

# Test data path
data_path = os.path.join(project_root, 'data', 'industrial_sensor_data_sample_clean.csv')
print("Data path:", data_path)
print("Data file exists:", os.path.exists(data_path))

# Test loading data
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
    print(f"Data loaded: {data.shape[0]:,} records, {data.shape[1]} columns")
else:
    print("Data file not found!")
