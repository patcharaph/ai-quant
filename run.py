"""
Simple launcher script for the AI Quant Stock Predictor
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'tensorflow', 
        'scikit-learn', 'plotly', 'yfinance', 'ta'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 Starting AI Quant Stock Predictor...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Exiting.")
        return
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please ensure you're in the correct directory.")
        return
    
    print("✅ All checks passed!")
    print("🌐 Launching web application...")
    print("=" * 50)
    
    # Ask user which app to run
    print("\nChoose application to run:")
    print("1. Main AI Quant App (app.py)")
    print("2. UI Demo (ui_demo.py)")
    print("3. Black Theme Demo (black_theme_demo.py)")
    print("4. Mock Data Test (test_mock.py)")
    print("5. Demo with Real Data (demo.py)")
    
    choice = input("\nEnter choice (1-5) [default: 1]: ").strip()
    
    if choice == "2":
        app_file = "ui_demo.py"
        print("🎨 Launching UI Demo...")
    elif choice == "3":
        app_file = "black_theme_demo.py"
        print("🌙 Launching Black Theme Demo...")
    elif choice == "4":
        print("🧪 Running Mock Data Test...")
        try:
            subprocess.run([sys.executable, 'test_mock.py'])
        except KeyboardInterrupt:
            print("\n👋 Test stopped by user.")
        except Exception as e:
            print(f"❌ Error running test: {e}")
        return
    elif choice == "5":
        print("📊 Running Demo with Real Data...")
        try:
            subprocess.run([sys.executable, 'demo.py'])
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user.")
        except Exception as e:
            print(f"❌ Error running demo: {e}")
        return
    else:
        app_file = "app.py"
        print("🤖 Launching Main AI Quant App...")
    
    try:
        # Launch Streamlit app
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_file])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main()
