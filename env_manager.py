"""
Environment Variable Manager for AI Quant Stock Prediction System

This module provides a centralized way to manage environment variables across
different deployment environments:

- Local: Uses .env file (ignored by .gitignore)
- Repository: Uses .env.example as template
- Production: Uses platform-provided environment variables

Usage:
    from env_manager import get_env_config
    
    config = get_env_config()
    api_key = config.OPENROUTER_API_KEY
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class EnvironmentConfig:
    """Configuration class for environment variables with type hints and defaults."""
    
    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str = 'openrouter/auto'
    OPENROUTER_BASE_URL: str = 'https://openrouter.ai/api/v1'
    
    # Optional model settings
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.OPENROUTER_API_KEY or self.OPENROUTER_API_KEY == 'your_openrouter_api_key_here':
            raise ValueError(
                "OPENROUTER_API_KEY is required. Please set it in your .env file or environment variables."
            )


class EnvironmentManager:
    """Manages environment variable loading with fallback support."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the environment manager.
        
        Args:
            project_root: Path to project root. If None, uses current working directory.
        """
        self.project_root = project_root or Path.cwd()
        self.env_file = self.project_root / '.env'
        self.env_example_file = self.project_root / '.env.example'
    
    def load_environment(self) -> Dict[str, str]:
        """
        Load environment variables with the following priority:
        1. System environment variables (highest priority)
        2. .env file (for local development)
        3. .env.example file (as fallback template)
        
        Returns:
            Dictionary of environment variables
        """
        env_vars = {}
        
        # First, try to load from .env file (for local development)
        if self.env_file.exists():
            load_dotenv(self.env_file)
            print(f"✓ Loaded environment from {self.env_file}")
        elif self.env_example_file.exists():
            # Load from .env.example as fallback
            load_dotenv(self.env_example_file)
            print(f"⚠️  Using .env.example as fallback. Consider creating .env file for local development.")
        else:
            print("⚠️  No .env or .env.example file found. Using system environment variables only.")
        
        # Load all relevant environment variables
        env_keys = [
            'OPENROUTER_API_KEY',
            'OPENROUTER_MODEL', 
            'OPENROUTER_BASE_URL',
            'MAX_TOKENS',
            'TEMPERATURE'
        ]
        
        for key in env_keys:
            value = os.getenv(key)
            if value is not None:
                env_vars[key] = value
        
        return env_vars
    
    def get_config(self) -> EnvironmentConfig:
        """
        Get environment configuration with proper type conversion and validation.
        
        Returns:
            EnvironmentConfig object with validated settings
            
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        env_vars = self.load_environment()
        
        # Type conversion and validation
        config_data = {}
        
        # Required string fields
        config_data['OPENROUTER_API_KEY'] = env_vars.get('OPENROUTER_API_KEY', 'your_openrouter_api_key_here')
        config_data['OPENROUTER_MODEL'] = env_vars.get('OPENROUTER_MODEL', 'openrouter/auto')
        config_data['OPENROUTER_BASE_URL'] = env_vars.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        
        # Optional numeric fields with type conversion
        try:
            config_data['MAX_TOKENS'] = int(env_vars.get('MAX_TOKENS', '500'))
        except ValueError:
            config_data['MAX_TOKENS'] = 500
            
        try:
            config_data['TEMPERATURE'] = float(env_vars.get('TEMPERATURE', '0.7'))
        except ValueError:
            config_data['TEMPERATURE'] = 0.7
        
        return EnvironmentConfig(**config_data)
    
    def create_env_file_from_example(self) -> bool:
        """
        Create .env file from .env.example template.
        
        Returns:
            True if file was created, False if .env already exists
        """
        if self.env_file.exists():
            print(f"✓ .env file already exists at {self.env_file}")
            return False
            
        if not self.env_example_file.exists():
            print(f"❌ .env.example file not found at {self.env_example_file}")
            return False
        
        # Copy .env.example to .env
        with open(self.env_example_file, 'r', encoding='utf-8') as src:
            content = src.read()
        
        with open(self.env_file, 'w', encoding='utf-8') as dst:
            dst.write(content)
        
        print(f"✓ Created .env file from .env.example at {self.env_file}")
        print("⚠️  Please edit .env file with your actual API keys before running the application.")
        return True


# Global environment manager instance
_env_manager = EnvironmentManager()


def get_env_config() -> EnvironmentConfig:
    """
    Get the current environment configuration.
    
    This is the main function to use throughout the application.
    
    Returns:
        EnvironmentConfig object with current environment settings
    """
    return _env_manager.get_config()


def create_local_env() -> bool:
    """
    Create .env file from .env.example template for local development.
    
    Returns:
        True if file was created, False if .env already exists
    """
    return _env_manager.create_env_file_from_example()


def get_environment_info() -> Dict[str, Any]:
    """
    Get information about the current environment setup.
    
    Returns:
        Dictionary with environment information
    """
    return {
        'env_file_exists': _env_manager.env_file.exists(),
        'env_example_exists': _env_manager.env_example_file.exists(),
        'project_root': str(_env_manager.project_root),
        'env_file_path': str(_env_manager.env_file),
        'env_example_path': str(_env_manager.env_example_file)
    }


if __name__ == "__main__":
    """Command-line interface for environment management."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-env":
        create_local_env()
    elif len(sys.argv) > 1 and sys.argv[1] == "info":
        info = get_environment_info()
        print("Environment Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        try:
            config = get_env_config()
            print("✓ Environment configuration loaded successfully:")
            print(f"  Model: {config.OPENROUTER_MODEL}")
            print(f"  Base URL: {config.OPENROUTER_BASE_URL}")
            print(f"  Max Tokens: {config.MAX_TOKENS}")
            print(f"  Temperature: {config.TEMPERATURE}")
            print(f"  API Key: {'*' * 8 + config.OPENROUTER_API_KEY[-4:] if config.OPENROUTER_API_KEY else 'Not set'}")
        except ValueError as e:
            print(f"❌ Environment configuration error: {e}")
            print("\nTo fix this:")
            print("1. Run: python env_manager.py create-env")
            print("2. Edit the .env file with your actual API keys")
            print("3. Run the application again")
