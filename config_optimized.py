"""
Optimized Configuration Manager for AI Quant Stock Prediction System

This module provides centralized configuration management with performance optimizations,
caching, and environment-specific settings.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class OptimizedConfig:
    """Optimized configuration manager with caching and validation"""
    
    def __init__(self, config_file: str = "config.py", env_file: str = ".env"):
        self.config_file = Path(config_file)
        self.env_file = Path(env_file)
        self._config_cache = {}
        self._env_cache = {}
        
        # Load configurations
        self._load_config()
        self._load_env()
    
    @lru_cache(maxsize=128)
    def get(self, key: str, default: Any = None, section: str = None) -> Any:
        """
        Get configuration value with caching
        
        Args:
            key: Configuration key
            default: Default value if key not found
            section: Configuration section (e.g., 'LSTM_CONFIG')
            
        Returns:
            Configuration value
        """
        if section:
            return self._config_cache.get(section, {}).get(key, default)
        return self._config_cache.get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config_cache.get(section, {})
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable with caching"""
        if key not in self._env_cache:
            self._env_cache[key] = os.getenv(key, default)
        return self._env_cache[key]
    
    def _load_config(self):
        """Load configuration from file without holding file locks (Windows-safe)"""
        try:
            if self.config_file.exists():
                # Read and exec into a temporary namespace to avoid import locks
                namespace: Dict[str, Any] = {}
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                exec(compile(code, str(self.config_file), 'exec'), {}, namespace)

                # Extract configuration dictionaries (UPPERCASE only)
                for attr_name, attr_value in namespace.items():
                    if isinstance(attr_name, str) and attr_name.isupper() and isinstance(attr_value, dict):
                        self._config_cache[attr_name] = attr_value

                logger.info(f"✅ Configuration loaded from {self.config_file}")
            else:
                logger.warning(f"⚠️  Configuration file {self.config_file} not found")

        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
    
    def _load_env(self):
        """Load environment variables"""
        try:
            if self.env_file.exists():
                with open(self.env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            self._env_cache[key.strip()] = value.strip()
                
                logger.info(f"✅ Environment variables loaded from {self.env_file}")
            else:
                logger.warning(f"⚠️  Environment file {self.env_file} not found")
                
        except Exception as e:
            logger.error(f"❌ Failed to load environment variables: {e}")
    
    def get_optimized_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get optimized model configuration"""
        base_config = self.get_section(f"{model_type.upper()}_CONFIG")
        
        # Apply optimizations based on available resources
        optimized_config = base_config.copy()
        
        # Optimize for available memory
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            
            if available_memory < 4:
                # Low memory optimization
                optimized_config['batch_size'] = min(32, optimized_config.get('batch_size', 64))
                optimized_config['hidden_units'] = min(32, optimized_config.get('hidden_units', 64))
            elif available_memory < 8:
                # Medium memory optimization
                optimized_config['batch_size'] = min(64, optimized_config.get('batch_size', 64))
            # High memory - use default values
                
        except ImportError:
            # psutil not available, use conservative defaults
            optimized_config['batch_size'] = 32
            optimized_config['hidden_units'] = 32
        
        return optimized_config
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get optimized data configuration"""
        data_config = self.get_section('DATA_CONFIG')
        
        # Optimize based on available CPU cores
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            
            # Adjust parallel processing settings
            data_config['n_jobs'] = min(4, cpu_count)
            data_config['chunk_size'] = max(1000, 10000 // cpu_count)
            
        except Exception:
            data_config['n_jobs'] = 1
            data_config['chunk_size'] = 1000
        
        return data_config
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration"""
        return {
            'use_gpu': self._check_gpu_availability(),
            'memory_growth': True,
            'mixed_precision': False,  # Can be enabled for newer GPUs
            'jit_compile': True,  # TensorFlow JIT compilation
            'parallel_processing': True,
            'cache_predictions': True,
            'max_cache_size': 1000
        }
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for TensorFlow"""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except Exception:
            return False
    
    def save_config(self, config_dict: Dict[str, Any], filename: str = None):
        """Save configuration to file"""
        filename = filename or "config_exported.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Configuration saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return issues"""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Check required sections
        required_sections = ['LSTM_CONFIG', 'TRANSFORMER_CONFIG', 'DATA_CONFIG']
        for section in required_sections:
            if section not in self._config_cache:
                issues['errors'].append(f"Missing required section: {section}")
        
        # Validate model configurations
        for model_type in ['LSTM', 'TRANSFORMER']:
            config = self.get_section(f"{model_type}_CONFIG")
            if config:
                if config.get('epochs', 0) <= 0:
                    issues['warnings'].append(f"{model_type}: Invalid epochs value")
                if config.get('learning_rate', 0) <= 0:
                    issues['warnings'].append(f"{model_type}: Invalid learning rate")
        
        return issues
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for debugging"""
        import platform
        import sys
        
        info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'node': platform.node(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version()
        }
        
        # Add package versions
        try:
            import pkg_resources
            packages = ['numpy', 'pandas', 'tensorflow', 'scikit-learn', 'streamlit']
            for package in packages:
                try:
                    version = pkg_resources.get_distribution(package).version
                    info[f'{package}_version'] = version
                except pkg_resources.DistributionNotFound:
                    info[f'{package}_version'] = 'Not installed'
        except ImportError:
            pass
        
        return info

# Global configuration instance
config = OptimizedConfig()

# Convenience functions
def get_config(key: str, default: Any = None, section: str = None) -> Any:
    """Get configuration value"""
    return config.get(key, default, section)

def get_model_config(model_type: str) -> Dict[str, Any]:
    """Get optimized model configuration"""
    return config.get_optimized_model_config(model_type)

def get_data_config() -> Dict[str, Any]:
    """Get optimized data configuration"""
    return config.get_data_config()

def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration"""
    return config.get_performance_config()

def validate_configuration() -> Dict[str, List[str]]:
    """Validate current configuration"""
    return config.validate_config()

def get_environment_info() -> Dict[str, Any]:
    """Get environment information"""
    return config.get_environment_info()
