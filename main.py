#!/usr/bin/env python3
"""
AI Quant Stock Predictor - Main Application Launcher

This is the optimized entry point for the AI Quant Stock Predictor application.
It handles environment setup, dependency checking, and graceful error handling.
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Optional

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    return True

def check_dependencies() -> List[str]:
    """Check for missing dependencies"""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'tensorflow', 
        'yfinance', 'streamlit', 'matplotlib', 'ta'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies(packages: List[str]) -> bool:
    """Install missing dependencies"""
    if not packages:
        return True
    
    print(f"📦 Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '--upgrade', '--quiet'
        ] + packages)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def setup_environment():
    """Setup environment and check configuration"""
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("🔧 Setting up environment file...")
        try:
            from env_manager import create_local_env
            create_local_env()
            print("✅ Environment file created from template")
            print("⚠️  Please edit .env file with your actual API keys")
        except Exception as e:
            print(f"❌ Failed to create environment file: {e}")
            return False
    
    return True

def optimize_imports():
    """Optimize imports for better performance"""
    # Set environment variables for optimization
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP conflicts
    os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL conflicts
    
    # Configure NumPy for better performance
    try:
        import numpy as np
        np.seterr(all='ignore')  # Ignore numerical warnings
    except ImportError:
        pass

def main():
    """Main application entry point"""
    print("🚀 AI Quant Stock Predictor")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"📦 Missing dependencies: {', '.join(missing_deps)}")
        if not install_dependencies(missing_deps):
            print("❌ Please install dependencies manually:")
            print(f"   pip install {' '.join(missing_deps)}")
            sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Optimize imports
    optimize_imports()
    
    # Launch Streamlit app
    print("🌐 Launching Streamlit application...")
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
