# 🏭 Industrial Sensor Data Streaming Generator

## 🎯 Project Overview

This package contains a **streaming data generator** for industrial sensor data. You'll use this to create realistic streaming data for building **real-time industrial IoT monitoring systems** and anomaly detection projects.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_streaming.txt
```

### 2. Start Streaming Data
```bash
# Basic console streaming (clean data - no anomaly labels)
python generate_flow.py --mode console --rate 1.0

# File streaming for analysis
python generate_flow.py --mode file --output my_stream.csv --rate 5.0

# Interactive mode for testing
python generate_flow.py --mode console --interactive --rate 1.0
```

### 3. Build Your Own Anomaly Detection System
Use the streaming data to build your own:
- Real-time anomaly detection algorithms
- Monitoring dashboards
- Data processing pipelines
- API endpoints for data ingestion


## 📁 Package Contents

### **Core Files**
- `generate_flow.py` - Main streaming data generator
- `setup.py` - Automated setup script

### **Configuration**
- `streaming_config.yaml` - Streaming configuration presets
- `kafka_config.yaml` - Kafka settings

### **Documentation**
- `STREAMING_GUIDE.md` - Comprehensive usage guide
- `requirements_streaming.txt` - Python dependencies

## 🎯 Key Features

### **Real-time Data Generation**
- **7 sensor types** with realistic industrial ranges
- **5 failure patterns** with gradual progression
- **Time-based variations** (daily/weekly cycles, seasonal trends)
- **Machine-specific characteristics** for multi-machine scenarios
- **Derived features** (rolling statistics, ratios, z-scores)

### **Clean Data for Learning**
- **No anomaly labels by default** - you must detect anomalies yourself!
- **Realistic failure patterns** - anomalies occur naturally during streaming
- **Consistent data structure** - same format as training datasets

### **Multiple Output Modes**
- **Console**: Real-time display for testing
- **File**: Continuous CSV/JSON file writing
- **API**: HTTP POST requests to endpoints
- **Kafka**: Message queue for distributed systems

## 🔧 Usage Examples

### **Basic Streaming**
```bash
# Console output at 1 Hz
python generate_flow.py --mode console --rate 1.0

# File output at 5 Hz
python generate_flow.py --mode file --output stream.csv --rate 5.0
```

### **Interactive Testing**
```bash
# Enable interactive controls
python generate_flow.py --mode console --interactive --rate 1.0

# Commands:
# 'a MACHINE_001 bearing_failure' - Inject anomaly
# 'c MACHINE_001' - Clear anomaly
# 'm MACHINE_001 30' - Start maintenance
# 's' - Show machine states
```

### **Kafka Integration**
```bash
# Stream to Kafka topic
python generate_flow.py --mode kafka --topic sensor-data --rate 10.0
```

### **API Integration**
```bash
# Stream to API endpoint
python generate_flow.py --mode api --url http://localhost:8000/api/data --rate 2.0
```

## 📊 Data Structure

### **Clean Data (Default)**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "machine_id": "MACHINE_001",
  "maintenance_status": "operational",
  "temperature": 72.5,
  "vibration": 4.2,
  "pressure": 5.8,
  "flow_rate": 195.3,
  "power_consumption": 48.7,
  "oil_level": 87.2,
  "bearing_temperature": 78.9,
  "hour": 10,
  "day_of_week": 0,
  "month": 1,
  "is_weekend": 0,
  "is_night_shift": 0,
  "temp_pressure_ratio": 12.5,
  "vibration_power_ratio": 0.086,
  "flow_pressure_product": 1132.74
}
```

## 🎓 Learning Objectives

By working with this project, you will learn:

1. **Time Series Analysis**
   - Real-time data preprocessing and feature engineering
   - Trend and seasonality detection in streaming data
   - Forecasting techniques for live data

2. **Anomaly Detection**
   - Multiple detection algorithms (Isolation Forest, LOF, One-Class SVM)
   - Performance evaluation and tuning
   - Real-time detection systems

3. **Stream Processing**
   - Kafka integration and message queues
   - Real-time data pipelines
   - Distributed processing concepts

4. **MLOps**
   - Model lifecycle management
   - Pipeline automation
   - Production deployment

5. **System Design**
   - Scalable architecture
   - Real-time processing
   - Monitoring and alerting

## 🚀 Project Ideas

1. **Real-time Monitoring Dashboard** - Build a live monitoring system
2. **Streaming Anomaly Detection** - Implement real-time anomaly detection
3. **Predictive Maintenance API** - Create an API for maintenance predictions
4. **Distributed Processing** - Build a Kafka-based processing pipeline
5. **ML Model Serving** - Deploy models for real-time predictions


## 🤝 Getting Help

1. **Read the documentation** - Start with `STREAMING_GUIDE.md`
2. **Experiment with configurations** - Use the presets in `config/streaming_config.yaml`
3. **Test your setup** - Use interactive mode to verify everything works

## 🎉 Ready to Start?

1. Install dependencies: `pip install -r requirements_streaming.txt`
2. Start with console mode: `python generate_flow.py --mode console --rate 1.0`
3. Try interactive mode: `python generate_flow.py --mode console --interactive --rate 1.0`
4. Build your own anomaly detection system!

**Good luck with your industrial IoT monitoring project!** 🏭✨
