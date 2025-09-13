"""
Tests for data_loader.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_loader import DataLoader

class TestDataLoader:
    """Test cases for DataLoader class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.loader = DataLoader()
    
    def test_validate_symbol(self):
        """Test symbol validation and normalization"""
        # Test adding .BK suffix
        assert self.loader.validate_symbol('AAPL') == 'AAPL.BK'
        assert self.loader.validate_symbol('AAPL.BK') == 'AAPL.BK'
        
        # Test valid symbols
        assert 'AAPL.BK' in self.loader.valid_symbols
        assert 'SET.BK' in self.loader.valid_symbols
    
    def test_bangkok_timezone(self):
        """Test Bangkok timezone configuration"""
        assert self.loader.bangkok_tz.zone == 'Asia/Bangkok'
    
    @pytest.mark.slow
    def test_fetch_ohlcv_mock(self):
        """Test OHLCV data fetching with mock data"""
        # This would be a mock test in practice
        # For now, just test the method exists
        assert hasattr(self.loader, 'fetch_ohlcv')
    
    def test_data_validation(self):
        """Test data validation logic"""
        # Create mock data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test validation
        result = self.loader.validate_data(mock_data, min_years=0.1)
        assert result is True
