"""
Setup script for AI Quant Stock Predictor
"""

import os
import shutil
import subprocess
import sys

def setup_environment():
    """Setup the environment for the AI Quant Stock Predictor"""
    print("🚀 Setting up AI Quant Stock Predictor...")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        if os.path.exists('env_example.txt'):
            shutil.copy('env_example.txt', '.env')
            print("✅ Created .env file from template")
            print("📝 Please edit .env and add your OpenRouter API key")
        else:
            print("⚠️ env_example.txt not found")
    else:
        print("✅ .env file already exists")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", f"{python_version.major}.{python_version.minor}")
        return False
    else:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    
    # Install requirements
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import streamlit
        import pandas
        import numpy
        import tensorflow
        import plotly
        import yfinance
        import ta
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenRouter API key (optional)")
    print("2. Run: streamlit run app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\n💡 For help, see README.md")
    
    return True

if __name__ == "__main__":
    success = setup_environment()
    if not success:
        sys.exit(1)
