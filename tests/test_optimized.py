"""
Optimized Test Suite for AI Quant Stock Prediction System

This test suite focuses on performance, reliability, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import DataLoader
from featurizer import Featurizer
from models import LSTMModel, TransformerModel, EnhancedModelSelector
from backtester import Backtester, FeeCalculator, TradeLedger
from calibration import CalibrationMetrics, UncertaintyQuantifier
from baseline_models import NaiveModel, LinearBaseline, create_default_baseline_models
from performance_monitor import PerformanceMonitor, DataOptimizer, ModelOptimizer
from config_optimized import OptimizedConfig

class TestDataLoader:
    """Test data loading functionality"""
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        assert loader.bangkok_tz is not None
        assert len(loader.thai_symbols) > 0
        assert loader.max_retries > 0
    
    def test_symbol_validation(self):
        """Test symbol validation and mapping"""
        loader = DataLoader()
        
        # Test Thai symbol mapping
        assert loader.validate_symbol('PTT') == 'PTT.BK'
        assert loader.validate_symbol('SCB') == 'SCB.BK'
        assert loader.validate_symbol('SET') == '^SET.BK'
        
        # Test already mapped symbols
        assert loader.validate_symbol('PTT.BK') == 'PTT.BK'
        assert loader.validate_symbol('^SET.BK') == '^SET.BK'
    
    @patch('yfinance.Ticker')
    def test_fetch_ohlcv_mock(self, mock_ticker):
        """Test data fetching with mocked yfinance"""
        # Setup mock
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_ticker.return_value.history.return_value = mock_data
        
        loader = DataLoader()
        result = loader.fetch_ohlcv('PTT', '2023-01-01', '2023-01-03')
        
        assert not result.empty
        assert 'close' in result.columns
        assert result.attrs['timezone'] == 'Asia/Bangkok'
    
    def test_data_validation(self):
        """Test data validation"""
        loader = DataLoader()
        
        # Create test data
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        validation = loader.validate_data(data)
        assert 'is_valid' in validation
        assert 'warnings' in validation

class TestFeaturizer:
    """Test feature engineering"""
    
    def test_featurizer_initialization(self):
        """Test Featurizer initialization"""
        featurizer = Featurizer()
        assert featurizer.scaler is not None
        assert 'scaler_fitted_on_test' in featurizer.data_leakage_checks
    
    def test_create_base_features(self):
        """Test base feature creation"""
        featurizer = Featurizer()
        
        # Create sample data
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [103, 104, 105, 106, 107],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        features = featurizer.create_base_features(data)
        
        # Check that features were created
        assert 'log_return' in features.columns
        assert 'rsi' in features.columns
        assert 'macd' in features.columns
        assert 'atr' in features.columns
    
    def test_target_creation(self):
        """Test target variable creation"""
        featurizer = Featurizer()
        
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        # Test price target
        result_price = featurizer.create_target_variable(data, 2, 'price')
        assert 'target' in result_price.columns
        assert result_price.attrs['target_type'] == 'price'
        
        # Test return target
        result_return = featurizer.create_target_variable(data, 2, 'return')
        assert 'target' in result_return.columns
        assert result_return.attrs['target_type'] == 'return'
        
        # Test hit probability target
        result_hit = featurizer.create_target_variable(data, 2, 'hit_probability', 3.0)
        assert 'target' in result_hit.columns
        assert result_hit.attrs['target_type'] == 'hit_probability'
    
    def test_data_leakage_prevention(self):
        """Test data leakage prevention"""
        featurizer = Featurizer()
        
        # Create sample data
        data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 105,
            'low': np.random.randn(100) + 95,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 2000, 100)
        })
        
        # Test supervised learning with walk-forward
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = featurizer.make_supervised(
            data, lookback_window=10, horizon_days=5, target_type='return', use_walk_forward=True
        )
        
        # Verify no data leakage
        assert metadata['data_leakage_checks']['scaler_fitted_on_test'] == False
        assert metadata['split_type'] == 'walk_forward'

class TestModels:
    """Test model functionality"""
    
    def test_lstm_model_initialization(self):
        """Test LSTM model initialization"""
        model = LSTMModel({'hidden_units': 32, 'layers': 1})
        assert model.config['hidden_units'] == 32
        assert model.config['layers'] == 1
    
    def test_transformer_model_initialization(self):
        """Test Transformer model initialization"""
        model = TransformerModel({'d_model': 32, 'n_heads': 2})
        assert model.config['d_model'] == 32
        assert model.config['n_heads'] == 2
    
    def test_enhanced_model_selector(self):
        """Test enhanced model selector"""
        selector = EnhancedModelSelector()
        assert selector.logger is not None
        assert selector.baseline_results == {}
    
    @patch('tensorflow.keras.models.Sequential')
    def test_lstm_build_model(self, mock_sequential):
        """Test LSTM model building"""
        model = LSTMModel({'hidden_units': 32, 'layers': 1})
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        
        model.build_model((10, 5))
        
        # Verify model was built
        assert mock_model.add.called
        assert mock_model.compile.called

class TestBacktester:
    """Test backtesting functionality"""
    
    def test_fee_calculator(self):
        """Test fee calculation"""
        calculator = FeeCalculator()
        
        fees = calculator.calculate_fees(10000, 'buy', 'PTT.BK')
        
        assert 'total_fees' in fees
        assert 'brokerage_fee' in fees
        assert 'vat' in fees
        assert fees['trade_value'] == 10000
    
    def test_trade_ledger(self):
        """Test trade ledger functionality"""
        ledger = TradeLedger()
        
        # Add a trade
        trade_data = {
            'symbol': 'PTT.BK',
            'action': 'buy',
            'quantity': 100,
            'price': 50.0,
            'total_cost': 5000.0,
            'fees': 7.5,
            'date': '2023-01-01'
        }
        
        ledger.add_trade(trade_data)
        
        assert len(ledger.trades) == 1
        assert ledger.trades[0]['trade_id'] == 1
        assert ledger.positions['PTT.BK']['quantity'] == 100
    
    def test_backtester_initialization(self):
        """Test backtester initialization"""
        backtester = Backtester()
        assert backtester.fee_calculator is not None
        assert backtester.trade_ledger is not None

class TestCalibration:
    """Test calibration functionality"""
    
    def test_calibration_metrics(self):
        """Test calibration metrics calculation"""
        metrics = CalibrationMetrics()
        
        # Test data
        lower_bounds = np.array([1, 2, 3, 4, 5])
        upper_bounds = np.array([3, 4, 5, 6, 7])
        actual_values = np.array([2, 3, 4, 5, 6])
        
        picp = metrics.calculate_picp(lower_bounds, upper_bounds, actual_values)
        pinaw = metrics.calculate_pinaw(lower_bounds, upper_bounds, actual_values)
        
        assert 0 <= picp <= 1
        assert pinaw >= 0
    
    def test_uncertainty_quantifier(self):
        """Test uncertainty quantification"""
        quantifier = UncertaintyQuantifier()
        
        # Test evaluation
        lower_bounds = np.array([1, 2, 3])
        upper_bounds = np.array([3, 4, 5])
        actual_values = np.array([2, 3, 4])
        
        evaluation = quantifier.evaluate_uncertainty(lower_bounds, upper_bounds, actual_values)
        
        assert 'picp' in evaluation
        assert 'pinaw' in evaluation
        assert 'reliability_score' in evaluation

class TestBaselineModels:
    """Test baseline models"""
    
    def test_naive_model(self):
        """Test naive baseline model"""
        model = NaiveModel()
        
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert all(p == y[-1] for p in predictions)
    
    def test_linear_baseline(self):
        """Test linear baseline model"""
        model = LinearBaseline()
        
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert model.model is not None
    
    def test_baseline_model_selector(self):
        """Test baseline model selector"""
        selector = create_default_baseline_models()
        
        assert len(selector.models) > 0
        assert 'Naive' in selector.models
        assert 'Linear' in selector.models

class TestPerformanceMonitor:
    """Test performance monitoring"""
    
    def test_performance_monitor(self):
        """Test performance monitor functionality"""
        monitor = PerformanceMonitor()
        
        # Test timing
        monitor.start_timer('test_operation')
        time.sleep(0.01)  # Small delay
        duration = monitor.end_timer('test_operation')
        
        assert duration > 0
        assert 'test_operation' in monitor.metrics
    
    def test_data_optimizer(self):
        """Test data optimization"""
        # Create test DataFrame
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'cat_col': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        optimized_df = DataOptimizer.optimize_dataframe(df)
        
        assert len(optimized_df) == len(df)
        assert optimized_df['int_col'].dtype in [np.uint8, np.uint16, np.uint32]
        assert optimized_df['cat_col'].dtype.name == 'category'

class TestConfigOptimized:
    """Test optimized configuration"""
    
    def test_config_initialization(self):
        """Test configuration initialization"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
LSTM_CONFIG = {'hidden_units': 64, 'epochs': 10}
DATA_CONFIG = {'lookback_window': 60}
""")
            f.flush()
            
            config = OptimizedConfig(f.name)
            
            assert config.get('hidden_units', section='LSTM_CONFIG') == 64
            assert config.get('lookback_window', section='DATA_CONFIG') == 60
            
            os.unlink(f.name)
    
    def test_environment_info(self):
        """Test environment information gathering"""
        config = OptimizedConfig()
        env_info = config.get_environment_info()
        
        assert 'python_version' in env_info
        assert 'platform' in env_info

# Performance tests
@pytest.mark.slow
class TestPerformance:
    """Performance-focused tests"""
    
    def test_large_dataframe_optimization(self):
        """Test optimization with large DataFrame"""
        # Create large DataFrame
        df = pd.DataFrame({
            'col1': np.random.randint(0, 1000, 100000),
            'col2': np.random.randn(100000),
            'col3': np.random.choice(['A', 'B', 'C', 'D'], 100000)
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = DataOptimizer.optimize_dataframe(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Should use less memory
        assert optimized_memory < original_memory
    
    def test_model_training_performance(self):
        """Test model training performance"""
        from performance_monitor import time_function
        
        @time_function
        def dummy_training():
            # Simulate training
            time.sleep(0.1)
            return "trained"
        
        result = dummy_training()
        assert result == "trained"

# Integration tests
@pytest.mark.integration
class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline"""
        # This would test the full pipeline from data loading to prediction
        # For now, just test that components can be imported and initialized
        from data_loader import DataLoader
        from featurizer import Featurizer
        from models import LSTMModel
        
        loader = DataLoader()
        featurizer = Featurizer()
        model = LSTMModel()
        
        assert loader is not None
        assert featurizer is not None
        assert model is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
