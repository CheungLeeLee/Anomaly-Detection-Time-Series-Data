# 📁 Project File Index

## 🎯 Quick Access to Key Files

### ⭐ **Most Important Files**
- **`scripts/working_ml_detector.py`** - Main ML implementation (run this first!)
- **`results/isolation_forest_results.csv`** - Anomaly detection results
- **`reports/ML_ANOMALY_DETECTION_REPORT.md`** - Comprehensive analysis report
- **`README.md`** - Project overview and quick start guide

### 📊 **Data Files**
- `data/industrial_sensor_data_sample_clean.csv` - Main dataset (8,640 records)
- `data/industrial_sensor_data_full_clean.csv` - Full dataset (larger file)

### 🐍 **Scripts by Purpose**

#### **Core ML Implementation**
- `scripts/working_ml_detector.py` - ⭐ **START HERE** - Main Isolation Forest
- `scripts/ml_model_comparison.py` - Compare all 4 ML models
- `scripts/isolation_forest_detector.py` - Advanced Isolation Forest analysis

#### **Analysis & Testing**
- `scripts/simple_anomaly_analysis.py` - Statistical baseline methods
- `scripts/anomaly_detection_analysis.py` - Comprehensive analysis (needs packages)
- `scripts/ml_anomaly_detection.py` - Full ML pipeline
- `scripts/test_ml.py` - Simple test script

### 📈 **Results & Outputs**
- `results/isolation_forest_results.csv` - ⭐ **Main Results** - 864 anomalies detected
- `results/ml_model_comparison_results.csv` - Multi-model comparison data

### 📋 **Reports & Documentation**
- `reports/ML_ANOMALY_DETECTION_REPORT.md` - ⭐ **Main Report** - ML analysis
- `reports/ANOMALY_DETECTION_REPORT.md` - Statistical analysis report
- `docs/ml_comparison_output.txt` - Model comparison results
- `docs/requirements.txt` - Python dependencies

### 🖼️ **Visualizations**
- `docs/anomaly_detection_visualization.png` - Data visualization
- `docs/anomaly_score_analysis.png` - Score analysis plots

## 🚀 **Quick Commands**

### Run Main Analysis
```bash
# From project root directory
python scripts/working_ml_detector.py
```

### Compare All Models
```bash
# From project root directory
python scripts/ml_model_comparison.py
```

**Important**: Always run scripts from the project root directory (`Anomaly Detection & Time Series Data/`). The scripts are configured to automatically find data files in `data/` folder and save results to `results/` folder.

### Install Dependencies
```bash
pip install -r docs/requirements.txt
```

## 📊 **Key Numbers**
- **Total Records**: 8,640
- **Anomalies Detected**: 864 (10%)
- **Models Trained**: 4 ML algorithms
- **Best Model**: Isolation Forest
- **Training Time**: < 1 second

## 🎯 **What Each Folder Contains**

| Folder | Purpose | Key Files |
|--------|---------|-----------|
| `data/` | Raw datasets | Sample & full CSV files |
| `scripts/` | Python code | All analysis scripts |
| `results/` | Model outputs | Anomaly detection results |
| `reports/` | Documentation | Analysis reports |
| `docs/` | Supporting files | Requirements, logs, images |
| `models/` | Saved models | (Future use) |
| `streaming_package/` | Real-time components | Kafka streaming setup |

## 🔍 **File Naming Convention**

- **`*_results.csv`** - Model prediction outputs
- **`*_detector.py`** - ML model implementations  
- **`*_analysis.py`** - Data analysis scripts
- **`*_comparison.py`** - Model comparison scripts
- **`*_REPORT.md`** - Analysis reports
- **`*_output.txt`** - Execution logs

## 📞 **Need Help?**

1. **Start with**: `scripts/working_ml_detector.py`
2. **Read**: `reports/ML_ANOMALY_DETECTION_REPORT.md`
3. **Check results**: `results/isolation_forest_results.csv`
4. **Review**: `README.md` for full project overview
