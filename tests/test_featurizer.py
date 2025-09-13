"""
Tests for featurizer.py
"""

import pytest
import pandas as pd
import numpy as np
from featurizer import Featurizer

class TestFeaturizer:
    """Test cases for Featurizer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.featurizer = Featurizer()
    
    def test_create_base_features(self):
        """Test base feature creation"""
        # Create mock OHLCV data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test feature creation
        result = self.featurizer.create_base_features(mock_data)
        
        # Check that new features are created
        assert 'log_return' in result.columns
        assert 'price_change' in result.columns
        assert 'hl_spread' in result.columns
        assert 'volume_change' in result.columns
    
    def test_data_leakage_checks(self):
        """Test data leakage detection"""
        # Test initial state
        assert self.featurizer.data_leakage_checks['scaler_fitted_on_test'] is False
        assert self.featurizer.data_leakage_checks['future_data_in_features'] is False
        assert self.featurizer.data_leakage_checks['target_in_features'] is False
        assert self.featurizer.data_leakage_checks['lookback_window_validation'] is True
    
    def test_create_target_variable(self):
        """Test target variable creation"""
        # Create mock data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)
        
        # Test price target
        result_price = self.featurizer.create_target_variable(mock_data, 5, 'price')
        assert 'target' in result_price.columns
        assert self.featurizer.target_column == 'target'
        
        # Test return target
        result_return = self.featurizer.create_target_variable(mock_data, 5, 'return')
        assert 'target' in result_return.columns
