#!/usr/bin/env python3
"""
Setup script for Industrial Sensor Data Streaming Project

This script helps students set up their environment and test the streaming system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Please use Python 3.8 or higher.")
        return False

def install_dependencies():
    """Install required dependencies"""
    if not Path("requirements_streaming.txt").exists():
        print("❌ requirements_streaming.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements_streaming.txt",
        "Installing dependencies"
    )

def test_streaming_generator():
    """Test the streaming generator"""
    if not Path("generate_flow.py").exists():
        print("❌ generate_flow.py not found")
        return False
    
    return run_command(
        f"{sys.executable} generate_flow.py --mode console --rate 1.0 --duration 2 --quiet",
        "Testing streaming generator"
    )

def test_configuration():
    """Test configuration files"""
    config_files = ["streaming_config.yaml", "kafka_config.yaml"]
    
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"⚠️  Configuration file {config_file} not found")
        else:
            print(f"✅ Configuration file {config_file} found")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ["data", "logs", "output"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def main():
    """Main setup function"""
    print("="*60)
    print("🏭 INDUSTRIAL SENSOR DATA STREAMING PROJECT SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Dependency installation failed. You may need to install manually:")
        print("   pip install -r requirements_streaming.txt")
        return 1
    
    # Test streaming generator
    if not test_streaming_generator():
        print("\n⚠️  Streaming generator test failed. Check the error messages above.")
        return 1
    
    # Test examples
    if not test_examples():
        print("\n⚠️  Example scripts test failed. Check the error messages above.")
        return 1
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n🚀 You're ready to start! Try these commands:")
    print("\n1. Basic streaming:")
    print("   python src/generate_flow.py --mode console --rate 1.0")
    print("\n2. Interactive mode:")
    print("   python src/generate_flow.py --mode console --interactive --rate 1.0")
    print("\n3. File streaming:")
    print("   python src/generate_flow.py --mode file --output data/my_stream.csv --rate 5.0")
    print("\n4. Real-time dashboard:")
    print("   streamlit run examples/real_time_dashboard.py")
    print("\n📚 Read README.md and STREAMING_GUIDE.md for more information!")
    
    return 0

if __name__ == "__main__":
    exit(main())
