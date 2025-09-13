#!/usr/bin/env python3
"""
Optimized Setup Script for AI Quant Stock Predictor

This script handles the complete setup process including:
- Environment setup
- Dependency installation
- Configuration
- Testing
- Performance optimization
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
from typing import List, Dict, Any

class OptimizedSetup:
    """Handles optimized setup process"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        
    def check_requirements(self) -> bool:
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if self.python_version < (3, 8):
            print(f"❌ Python 3.8+ required, found {self.python_version.major}.{self.python_version.minor}")
            return False
        
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 4:
                print(f"⚠️  Low memory detected: {memory_gb:.1f}GB (recommended: 4GB+)")
            else:
                print(f"✅ Memory: {memory_gb:.1f}GB")
        except ImportError:
            print("⚠️  Cannot check memory (psutil not available)")
        
        # Check disk space
        try:
            import shutil
            free_space = shutil.disk_usage(self.project_root).free / (1024**3)
            if free_space < 2:
                print(f"⚠️  Low disk space: {free_space:.1f}GB (recommended: 2GB+)")
            else:
                print(f"✅ Disk space: {free_space:.1f}GB")
        except Exception:
            print("⚠️  Cannot check disk space")
        
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment"""
        print("🐍 Creating virtual environment...")
        
        if self.venv_path.exists():
            print("✅ Virtual environment already exists")
            return True
        
        try:
            subprocess.run([
                sys.executable, '-m', 'venv', str(self.venv_path)
            ], check=True)
            print("✅ Virtual environment created")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_pip_command(self) -> List[str]:
        """Get pip command for current platform"""
        if self.system == "windows":
            pip_cmd = [str(self.venv_path / "Scripts" / "pip")]
        else:
            pip_cmd = [str(self.venv_path / "bin" / "pip")]
        
        return pip_cmd
    
    def get_python_command(self) -> List[str]:
        """Get python command for current platform"""
        if self.system == "windows":
            python_cmd = [str(self.venv_path / "Scripts" / "python")]
        else:
            python_cmd = [str(self.venv_path / "bin" / "python")]
        
        return python_cmd
    
    def install_dependencies(self, dev: bool = False, full: bool = False) -> bool:
        """Install dependencies"""
        print("📦 Installing dependencies...")
        
        pip_cmd = self.get_pip_command()
        
        # Upgrade pip first
        try:
            subprocess.run(pip_cmd + ['install', '--upgrade', 'pip'], check=True)
        except subprocess.CalledProcessError:
            print("⚠️  Failed to upgrade pip, continuing...")
        
        # Install base requirements
        requirements_file = self.project_root / "requirements_optimized.txt"
        if requirements_file.exists():
            try:
                subprocess.run(pip_cmd + ['install', '-r', str(requirements_file)], check=True)
                print("✅ Base dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install base dependencies: {e}")
                return False
        
        # Install optional dependencies
        if dev or full:
            try:
                if dev:
                    subprocess.run(pip_cmd + ['install', '-e', '.[dev]'], check=True)
                    print("✅ Development dependencies installed")
                if full:
                    subprocess.run(pip_cmd + ['install', '-e', '.[full]'], check=True)
                    print("✅ Full dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Failed to install optional dependencies: {e}")
        
        return True
    
    def setup_environment(self) -> bool:
        """Setup environment configuration"""
        print("🔧 Setting up environment...")
        
        python_cmd = self.get_python_command()
        
        # Create .env file
        try:
            subprocess.run(python_cmd + ['env_manager.py', 'create-env'], check=True)
            print("✅ Environment file created")
        except subprocess.CalledProcessError:
            print("⚠️  Failed to create environment file")
        
        # Create necessary directories
        directories = ['data_cache', 'model_logs', 'llm_logs', 'tests/test_data']
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
        
        print("✅ Directories created")
        return True
    
    def setup_pre_commit(self) -> bool:
        """Setup pre-commit hooks"""
        print("🪝 Setting up pre-commit hooks...")
        
        python_cmd = self.get_python_command()
        
        try:
            subprocess.run(python_cmd + ['-m', 'pre_commit', 'install'], check=True)
            print("✅ Pre-commit hooks installed")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Pre-commit hooks not installed (optional)")
            return False
    
    def run_tests(self) -> bool:
        """Run test suite"""
        print("🧪 Running tests...")
        
        python_cmd = self.get_python_command()
        
        try:
            subprocess.run(python_cmd + ['-m', 'pytest', 'tests/', '-v'], check=True)
            print("✅ All tests passed")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Some tests failed (check output above)")
            return False
    
    def optimize_system(self) -> bool:
        """Apply system optimizations"""
        print("⚡ Applying system optimizations...")
        
        # Set environment variables for optimization
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        # Create optimization script
        optimization_script = self.project_root / "optimize_system.py"
        with open(optimization_script, 'w') as f:
            f.write("""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run optimizations
try:
    from performance_monitor import initialize_optimizations
    initialize_optimizations()
    print("✅ System optimizations applied")
except Exception as e:
    print(f"⚠️  Optimization warning: {e}")
""")
        
        python_cmd = self.get_python_command()
        try:
            subprocess.run(python_cmd + [str(optimization_script)], check=True)
            optimization_script.unlink()  # Clean up
            return True
        except subprocess.CalledProcessError:
            print("⚠️  System optimization failed")
            return False
    
    def create_launcher_scripts(self) -> bool:
        """Create platform-specific launcher scripts"""
        print("🚀 Creating launcher scripts...")
        
        # Windows batch file
        if self.system == "windows":
            launcher_script = self.project_root / "run.bat"
            with open(launcher_script, 'w') as f:
                f.write(f"""@echo off
cd /d "{self.project_root}"
call venv\\Scripts\\activate
python main.py
pause
""")
            launcher_script.chmod(0o755)
        
        # Unix shell script
        else:
            launcher_script = self.project_root / "run.sh"
            with open(launcher_script, 'w') as f:
                f.write(f"""#!/bin/bash
cd "{self.project_root}"
source venv/bin/activate
python main.py
""")
            launcher_script.chmod(0o755)
        
        print("✅ Launcher scripts created")
        return True
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("🎉 Setup completed successfully!")
        print("="*60)
        print("\n📋 Next steps:")
        print("1. Edit .env file with your API keys:")
        print("   - OPENROUTER_API_KEY=your_actual_api_key_here")
        print("\n2. Run the application:")
        if self.system == "windows":
            print("   - Double-click run.bat")
            print("   - Or: venv\\Scripts\\activate && python main.py")
        else:
            print("   - ./run.sh")
            print("   - Or: source venv/bin/activate && python main.py")
        print("\n3. Open your browser to http://localhost:8501")
        print("\n4. Start predicting Thai stocks! 🚀")
        print("\n📚 Documentation: README_OPTIMIZED.md")
        print("🐛 Issues: https://github.com/ai-quant/stock-predictor/issues")
        print("="*60)
    
    def run_setup(self, dev: bool = False, full: bool = False, skip_tests: bool = False):
        """Run complete setup process"""
        print("🚀 AI Quant Stock Predictor - Optimized Setup")
        print("=" * 50)
        
        # Check requirements
        if not self.check_requirements():
            print("❌ System requirements not met")
            sys.exit(1)
        
        # Create virtual environment
        if not self.create_virtual_environment():
            print("❌ Failed to create virtual environment")
            sys.exit(1)
        
        # Install dependencies
        if not self.install_dependencies(dev=dev, full=full):
            print("❌ Failed to install dependencies")
            sys.exit(1)
        
        # Setup environment
        if not self.setup_environment():
            print("❌ Failed to setup environment")
            sys.exit(1)
        
        # Setup pre-commit (optional)
        self.setup_pre_commit()
        
        # Run tests (optional)
        if not skip_tests:
            self.run_tests()
        
        # Apply optimizations
        self.optimize_system()
        
        # Create launcher scripts
        self.create_launcher_scripts()
        
        # Print next steps
        self.print_next_steps()

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="AI Quant Stock Predictor Setup")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--full", action="store_true", help="Install full dependencies")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    
    args = parser.parse_args()
    
    setup = OptimizedSetup()
    setup.run_setup(dev=args.dev, full=args.full, skip_tests=args.skip_tests)

if __name__ == "__main__":
    main()
