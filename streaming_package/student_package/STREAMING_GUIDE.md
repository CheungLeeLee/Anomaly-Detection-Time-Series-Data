# Industrial Sensor Data Streaming Guide

## 🎯 Overview

The `generate_flow.py` script provides real-time streaming simulation of industrial sensor data, complementing the existing batch dataset generator. This streaming component enables students to work with live data flows, build real-time monitoring systems, and implement streaming anomaly detection.

## 🚀 Quick Start

### Basic Usage

```bash
# Console output at 1 Hz
python src/generate_flow.py --mode console --rate 1.0

# File output at 5 Hz
python src/generate_flow.py --mode file --output data/stream.csv --rate 5.0

# Interactive mode for testing
python src/generate_flow.py --mode console --interactive --rate 1.0
```

### Using Configuration Files

```bash
# Load from YAML configuration
python src/generate_flow.py --config config/streaming_config.yaml

# Use preset configurations
python src/generate_flow.py --config config/streaming_config.yaml --preset development
```

## 📊 Features

### **Core Capabilities**
- **Real-time data streaming** with configurable rates (0.1 - 100+ Hz)
- **Multiple output modes**: Console, File, API, Kafka
- **Real-time anomaly injection** with realistic failure patterns
- **Maintenance simulation** with configurable intervals
- **Interactive controls** for manual testing
- **Consistent data structure** with existing datasets

### **Data Generation**
- **7 sensor types** with realistic industrial ranges
- **5 failure patterns** with gradual progression
- **Time-based variations** (daily/weekly cycles, seasonal trends)
- **Machine-specific characteristics** for multi-machine scenarios
- **Derived features** (rolling statistics, ratios, z-scores)
- **Clean data by default** - no anomaly labels for student projects
- **Optional labels** for instructor testing and grading

### **Output Formats**
- **JSON**: Structured data for APIs and databases
- **CSV**: Tabular format for analysis tools
- **Kafka**: Message queue for distributed systems
- **API**: HTTP POST requests to endpoints

## 🔧 Configuration

### **Command Line Options**

```bash
# Basic settings
--rate 1.0                    # Streaming rate in Hz
--duration 3600              # Duration in seconds (infinite if not specified)
--machines MACHINE_001 MACHINE_002  # Specific machines (all if not specified)

# Output settings
--mode console               # Output mode: console, file, api, kafka
--output data/stream.csv     # Output file path (for file mode)
--url http://localhost:8000/api/data  # API URL (for api mode)
--topic sensor-data          # Kafka topic (for kafka mode)

# Data settings
--no-anomalies              # Disable anomaly injection
--anomaly-rate 0.01         # Anomaly injection probability
--no-maintenance            # Disable maintenance simulation
--maintenance-interval 3600 # Maintenance interval in seconds

# Format settings
--format json               # Data format: json, csv
--no-derived-features       # Disable derived features
--no-labels                 # Disable anomaly labels (default: disabled for student projects)

# Interactive settings
--interactive               # Enable interactive controls
--quiet                     # Disable verbose output
```

### **YAML Configuration**

```yaml
streaming:
  rate: 1.0
  duration: null
  machines: null
  output_mode: "console"
  output_path: null
  api_url: null
  kafka_topic: null
  kafka_bootstrap_servers:
    - "localhost:9092"
  include_anomalies: true
  anomaly_injection_rate: 0.01
  maintenance_simulation: true
  maintenance_interval: 3600
  data_format: "json"
  include_derived_features: true
  include_anomaly_labels: true
  interactive_mode: false
  verbose: true
```

## 📈 Use Cases

### **1. Development and Testing**
```bash
# Quick testing with console output
python src/generate_flow.py --mode console --rate 0.5 --duration 300 --interactive

# File logging for analysis
python src/generate_flow.py --mode file --output data/test_stream.csv --rate 1.0 --duration 600
```

### **2. API Development**
```bash
# Stream to API endpoint
python src/generate_flow.py --mode api --url http://localhost:8000/api/data --rate 2.0

# Test API with anomalies
python src/generate_flow.py --mode api --url http://localhost:8000/api/data --rate 1.0 --anomaly-rate 0.05
```

### **3. Kafka Integration**
```bash
# Stream to Kafka topic
python src/generate_flow.py --mode kafka --topic sensor-data --rate 10.0

# High-frequency streaming
python src/generate_flow.py --mode kafka --topic sensor-data --rate 50.0 --no-anomalies
```

### **4. Production Simulation**
```bash
# Realistic production scenario
python src/generate_flow.py --mode kafka --topic sensor-data --rate 5.0 --anomaly-rate 0.001 --maintenance-interval 7200
```

## 🎮 Interactive Controls

When using `--interactive` mode, you can control the streaming in real-time:

```
🎮 Interactive Controls:
  'a <machine_id> [failure_type]' - Inject anomaly
  'c <machine_id>' - Clear anomaly
  'm <machine_id> [duration]' - Start maintenance
  's' - Show machine states
  'q' - Quit
  'h' - Show this help

> a MACHINE_001 bearing_failure
🚨 Injected bearing_failure anomaly for MACHINE_001

> s
📊 Machine States:
  MACHINE_001: ✅ Operational | 🚨 bearing_failure
  MACHINE_002: ✅ Operational | ✅ Normal
  MACHINE_003: 🔧 Maintenance | ✅ Normal
  MACHINE_004: ✅ Operational | ✅ Normal
  MACHINE_005: ✅ Operational | ✅ Normal

> c MACHINE_001
✅ Cleared anomaly for MACHINE_001
```

## 🔌 Integration Examples

### **1. Kafka Consumer**
```python
# examples/streaming_consumer.py
python examples/streaming_consumer.py --topic sensor-data --group-id my-consumer --anomaly-detection
```

### **2. Real-time Dashboard**
```python
# examples/real_time_dashboard.py
streamlit run examples/real_time_dashboard.py
```

### **3. Anomaly Detection**
```python
# examples/anomaly_detector.py
python examples/anomaly_detector.py --input kafka --topic sensor-data --algorithm isolation_forest
```

## 🎯 **Clean Data for Student Projects**

### **Why No Anomaly Labels by Default?**

The streaming generator is designed with **educational best practices** in mind:

- **Students should detect anomalies**, not receive pre-labeled data
- **Real-world scenarios** don't come with anomaly labels
- **Anomaly detection is the learning objective**, not just data processing
- **Instructors can enable labels** for testing and grading purposes

### **Two Modes of Operation**

#### **Student Mode (Default)**
```bash
# Clean data - no anomaly labels
python src/generate_flow.py --mode console --rate 1.0
# Output: Only sensor data and derived features
```

#### **Instructor Mode (For Testing/Grading)**
```bash
# Data with labels for instructor evaluation
python src/generate_flow.py --mode file --output instructor_data.csv --rate 1.0 --include-labels
# Output: Includes anomaly_label and anomaly_type columns
```

### **Configuration Presets**

- **`development`**: Clean data for student development
- **`production`**: Clean data for production simulation
- **`instructor_testing`**: Data with labels for instructor evaluation

## 📊 Data Structure

### **JSON Format (Student Projects - Clean Data)**
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

### **JSON Format (Instructor Testing - With Labels)**
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
  "anomaly_label": 0,
  "anomaly_type": "normal",
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

### **CSV Format (Student Projects - Clean Data)**
```csv
timestamp,machine_id,maintenance_status,temperature,vibration,pressure,flow_rate,power_consumption,oil_level,bearing_temperature,hour,day_of_week,month,is_weekend,is_night_shift,temp_pressure_ratio,vibration_power_ratio,flow_pressure_product
2024-01-15T10:30:00,MACHINE_001,operational,72.5,4.2,5.8,195.3,48.7,87.2,78.9,10,0,1,0,0,12.5,0.086,1132.74
```

### **CSV Format (Instructor Testing - With Labels)**
```csv
timestamp,machine_id,maintenance_status,temperature,vibration,pressure,flow_rate,power_consumption,oil_level,bearing_temperature,anomaly_label,anomaly_type,hour,day_of_week,month,is_weekend,is_night_shift,temp_pressure_ratio,vibration_power_ratio,flow_pressure_product
2024-01-15T10:30:00,MACHINE_001,operational,72.5,4.2,5.8,195.3,48.7,87.2,78.9,0,normal,10,0,1,0,0,12.5,0.086,1132.74
```

## 🏗️ Architecture

### **System Components**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Streaming Gen   │───▶│   Output Sink   │
│  (Config/CLI)   │    │  (generate_flow) │    │ (Console/File/  │
└─────────────────┘    └──────────────────┘    │  API/Kafka)     │
                                               └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │   Consumers     │
                                               │ (Dashboard/     │
                                               │  Detector/      │
                                               │  Analytics)     │
                                               └─────────────────┘
```

### **Data Flow**

1. **Configuration**: Load settings from CLI args or YAML file
2. **Initialization**: Setup output handlers and machine states
3. **Streaming Loop**: Generate data points at specified rate
4. **Anomaly Injection**: Background thread for automatic anomalies
5. **Maintenance Simulation**: Background thread for maintenance events
6. **Interactive Control**: User input thread for manual control
7. **Output**: Send data to configured output sink

## 🎓 Educational Applications

### **Beginner Level**
- **Real-time visualization** with Streamlit dashboards
- **Basic anomaly detection** using threshold methods
- **Data pipeline understanding** with file streaming

### **Intermediate Level**
- **Stream processing** with Kafka integration
- **ML-based anomaly detection** with scikit-learn
- **API development** for data ingestion

### **Advanced Level**
- **Distributed streaming** with multiple consumers
- **Real-time ML pipelines** with model serving
- **Production monitoring** with alerting systems

## 🔧 Troubleshooting

### **Common Issues**

1. **Kafka Connection Failed**
   ```bash
   # Check if Kafka is running
   docker ps | grep kafka
   
   # Start Kafka if needed
   docker-compose up -d kafka
   ```

2. **API Connection Failed**
   ```bash
   # Test API endpoint
   curl http://localhost:8000/api/data
   
   # Check if API server is running
   netstat -tlnp | grep 8000
   ```

3. **High Memory Usage**
   ```bash
   # Reduce buffer sizes
   python src/generate_flow.py --mode file --rate 0.1 --duration 300
   ```

4. **File Permission Issues**
   ```bash
   # Create output directory
   mkdir -p data
   chmod 755 data
   ```

### **Performance Tuning**

- **Rate Limiting**: Start with low rates (0.1-1.0 Hz) for testing
- **Buffer Management**: Use appropriate buffer sizes for your use case
- **Output Optimization**: File output is fastest, Kafka has overhead
- **Memory Usage**: Monitor memory usage for long-running streams

## 📚 Additional Resources

### **Related Scripts**
- `generate_industrial_dataset.py`: Batch dataset generation
- `streaming_consumer.py`: Kafka consumer example
- `real_time_dashboard.py`: Streamlit dashboard
- `anomaly_detector.py`: Real-time anomaly detection

### **Configuration Files**
- `config/streaming_config.yaml`: Default streaming configuration
- `config/kafka_config.yaml`: Kafka-specific settings

### **Documentation**
- `DATASET_SUMMARY.md`: Overview of generated datasets
- `FINAL_SUMMARY.md`: Complete project summary
- `README_dataset_generator.md`: Batch generator documentation

## 🚀 Next Steps

1. **Start with console mode** to understand the data structure
2. **Experiment with interactive controls** for manual testing
3. **Try different output modes** (file, API, Kafka)
4. **Build consumers** using the provided examples
5. **Create your own dashboards** and monitoring systems
6. **Implement custom anomaly detection** algorithms
7. **Scale to production** with proper monitoring and alerting

The streaming component provides a realistic foundation for building production-ready industrial IoT monitoring systems!
