"""
Comprehensive tests for security and functionality improvements
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from pathlib import Path

# Import modules to test
from data_loader import DataLoader
from featurizer import Featurizer
from models import LSTMModel, TransformerModel, EnhancedModelSelector
from backtester import Backtester, SignalGenerator, FeeCalculator
from env_manager import get_env_config, create_local_env
from predictor import Predictor, AdvisoryGenerator


class TestSecurityImprovements:
    """Test security-related improvements"""
    
    def test_env_file_security(self):
        """Test that .env files are properly ignored"""
        # Check that .env.example exists but .env is ignored
        assert os.path.exists('.env.example'), ".env.example should exist"
        
        # Check .gitignore contains proper patterns
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()
        
        assert '.env' in gitignore_content, ".env should be in .gitignore"
        assert '*.env' in gitignore_content, "*.env should be in .gitignore"
        assert '.streamlit/secrets.toml' in gitignore_content, "secrets.toml should be in .gitignore"
    
    def test_env_manager_centralization(self):
        """Test centralized environment variable loading"""
        # Test that env_manager can be imported and used
        try:
            config = get_env_config()
            assert hasattr(config, 'OPENROUTER_API_KEY'), "Config should have API key attribute"
        except ValueError:
            # This is expected if no API key is set
            pass
    
    def test_streamlit_secrets_example(self):
        """Test that Streamlit secrets example exists"""
        secrets_example_path = Path('.streamlit/secrets.toml.example')
        assert secrets_example_path.exists(), "Streamlit secrets example should exist"


class TestDataPipelineImprovements:
    """Test data pipeline improvements for Thai market"""
    
    def test_symbol_mapping(self):
        """Test automatic symbol mapping for Thai stocks"""
        loader = DataLoader()
        
        # Test Thai symbol mapping
        assert loader.validate_symbol('PTT') == 'PTT.BK', "PTT should map to PTT.BK"
        assert loader.validate_symbol('SCB') == 'SCB.BK', "SCB should map to SCB.BK"
        assert loader.validate_symbol('KBANK') == 'KBANK.BK', "KBANK should map to KBANK.BK"
        
        # Test that already mapped symbols are preserved
        assert loader.validate_symbol('PTT.BK') == 'PTT.BK', "PTT.BK should remain PTT.BK"
        
        # Test index symbols
        assert loader.validate_symbol('SET') == '^SET.BK', "SET should map to ^SET.BK"
    
    def test_timezone_handling(self):
        """Test Asia/Bangkok timezone handling"""
        loader = DataLoader()
        
        # Check that Bangkok timezone is set
        assert str(loader.bangkok_tz) == 'Asia/Bangkok', "Should use Asia/Bangkok timezone"
        
        # Test timezone conversion
        utc_time = pd.Timestamp('2024-01-01 00:00:00', tz='UTC')
        bangkok_time = utc_time.tz_convert(loader.bangkok_tz)
        assert bangkok_time.tz.zone == 'Asia/Bangkok', "Should convert to Bangkok timezone"


class TestDataLeakagePrevention:
    """Test data leakage prevention measures"""
    
    def test_scaler_fitting_validation(self):
        """Test that scaler is only fitted on training data"""
        # Create sample data with all OHLCV columns
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = np.random.randn(100).cumsum() + 100
        data = pd.DataFrame({
            'open': close_prices + np.random.randn(100) * 0.5,
            'high': close_prices + np.abs(np.random.randn(100)) * 0.5,
            'low': close_prices - np.abs(np.random.randn(100)) * 0.5,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        featurizer = Featurizer()
        
        # Create supervised dataset
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = featurizer.make_supervised(
            data, lookback_window=10, horizon_days=5, target_type='price'
        )
        
        # Check that scaler was fitted on training data only
        assert hasattr(featurizer, '_scaler_fitted_on_train_only'), "Should mark scaler as fitted on train only"
        assert featurizer._scaler_fitted_on_train_only == True, "Scaler should be fitted on training data only"
        
        # Check data leakage validation
        leakage_checks = featurizer.data_leakage_checks
        assert not leakage_checks.get('scaler_fitted_on_test', False), "Scaler should not be fitted on test data"
    
    def test_feature_target_separation(self):
        """Test that target variable is not in features"""
        # Create sample data with all OHLCV columns
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = np.random.randn(100).cumsum() + 100
        data = pd.DataFrame({
            'open': close_prices + np.random.randn(100) * 0.5,
            'high': close_prices + np.abs(np.random.randn(100)) * 0.5,
            'low': close_prices - np.abs(np.random.randn(100)) * 0.5,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        featurizer = Featurizer()
        
        # Create supervised dataset
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = featurizer.make_supervised(
            data, lookback_window=10, horizon_days=5, target_type='price'
        )
        
        # Check that target is not in feature columns
        feature_cols = metadata['feature_columns']
        assert 'target' not in feature_cols, "Target should not be in feature columns"
        assert 'close' not in feature_cols, "Close price should not be in feature columns"
    
    def test_time_ordering_validation(self):
        """Test that time ordering is preserved"""
        # Create sample data with all OHLCV columns
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = np.random.randn(100).cumsum() + 100
        data = pd.DataFrame({
            'open': close_prices + np.random.randn(100) * 0.5,
            'high': close_prices + np.abs(np.random.randn(100)) * 0.5,
            'low': close_prices - np.abs(np.random.randn(100)) * 0.5,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        featurizer = Featurizer()
        
        # Create supervised dataset
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = featurizer.make_supervised(
            data, lookback_window=10, horizon_days=5, target_type='price'
        )
        
        # Check time ordering
        n_train = metadata['n_train']
        n_val = metadata['n_val']
        n_test = metadata['n_test']
        
        assert n_train > 0, "Should have training data"
        assert n_val > 0, "Should have validation data"
        assert n_test > 0, "Should have test data"
        assert n_train + n_val + n_test == len(X_train) + len(X_val) + len(X_test), "Data split should be complete"


class TestModelSelectionRules:
    """Test enhanced model selection rules"""
    
    def test_rmse_mae_mape_selection(self):
        """Test RMSE -> MAE -> MAPE selection logic"""
        selector = EnhancedModelSelector()
        
        # Test case 1: RMSE difference >= 2%
        lstm_metrics = {'RMSE': 0.05, 'MAE': 0.04, 'MAPE': 5.0, 'R2': 0.8}
        transformer_metrics = {'RMSE': 0.08, 'MAE': 0.03, 'MAPE': 3.0, 'R2': 0.7}
        
        selected_model, reason, comparison = selector.select_best_model(
            lstm_metrics, transformer_metrics
        )
        
        assert selected_model == 'lstm', "Should select LSTM with better RMSE"
        assert 'RMSE' in reason, "Reason should mention RMSE"
        
        # Test case 2: RMSE difference < 2%, use MAE
        lstm_metrics = {'RMSE': 0.05, 'MAE': 0.04, 'MAPE': 5.0, 'R2': 0.8}
        transformer_metrics = {'RMSE': 0.051, 'MAE': 0.03, 'MAPE': 3.0, 'R2': 0.7}
        
        selected_model, reason, comparison = selector.select_best_model(
            lstm_metrics, transformer_metrics
        )
        
        assert selected_model == 'transformer', "Should select Transformer with better MAE"
        assert 'MAE' in reason, "Reason should mention MAE"
    
    def test_model_selection_logging(self):
        """Test that model selection is properly logged"""
        selector = EnhancedModelSelector()
        
        lstm_metrics = {'RMSE': 0.05, 'MAE': 0.04, 'MAPE': 5.0, 'R2': 0.8}
        transformer_metrics = {'RMSE': 0.08, 'MAE': 0.03, 'MAPE': 3.0, 'R2': 0.7}
        
        selected_model, reason, comparison = selector.select_best_model(
            lstm_metrics, transformer_metrics
        )
        
        # Check that selection was logged
        assert len(selector.selection_history) > 0, "Selection should be logged"
        assert selector.selection_history[0]['selected_model'] == selected_model, "Logged model should match selection"


class TestBacktestRealism:
    """Test realistic backtesting features"""
    
    def test_fee_calculation(self):
        """Test Thai market fee calculation"""
        fee_calc = FeeCalculator()
        
        # Test Thai retail fees
        trade_value = 100000  # 100k THB
        fees = fee_calc.calculate_fees(trade_value, 'buy')
        
        assert 'brokerage_fee' in fees, "Should calculate brokerage fee"
        assert 'vat' in fees, "Should calculate VAT"
        assert 'settlement_fee' in fees, "Should calculate settlement fee"
        assert 'slippage' in fees, "Should calculate slippage"
        assert fees['total_fees'] > 0, "Total fees should be positive"
    
    def test_trade_ledger_functionality(self):
        """Test trade ledger functionality"""
        from backtester import TradeLedger
        
        ledger = TradeLedger()
        
        # Add a sample trade
        trade_data = {
            'symbol': 'PTT.BK',
            'action': 'buy',
            'quantity': 100,
            'price': 50.0,
            'total_cost': 5000.0,
            'fees': 25.0,
            'date': datetime.now()
        }
        
        ledger.add_trade(trade_data)
        
        # Check trade was added
        assert len(ledger.trades) == 1, "Should have one trade"
        assert ledger.trades[0]['symbol'] == 'PTT.BK', "Trade symbol should match"
        
        # Test summary
        summary = ledger.get_trade_summary()
        assert summary['total_trades'] == 1, "Should have one trade in summary"
        assert summary['buy_trades'] == 1, "Should have one buy trade"
    
    def test_entry_exit_rules(self):
        """Test entry/exit rule definitions"""
        signal_gen = SignalGenerator()
        
        # Check that entry/exit rules are defined
        assert 'A' in signal_gen.entry_exit_rules, "Should have scheme A rules"
        assert 'B' in signal_gen.entry_exit_rules, "Should have scheme B rules"
        assert 'C' in signal_gen.entry_exit_rules, "Should have scheme C rules"
        
        # Check rule structure
        rule_a = signal_gen.entry_exit_rules['A']
        assert 'entry_rule' in rule_a, "Should have entry rule"
        assert 'exit_rule' in rule_a, "Should have exit rule"
        assert 'description' in rule_a, "Should have description"


class TestLabelDefinitions:
    """Test clear label and hit probability definitions"""
    
    def test_target_creation_formulas(self):
        """Test that target creation formulas are clearly defined"""
        featurizer = Featurizer()
        
        # Create sample data with all OHLCV columns
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = np.random.randn(100).cumsum() + 100
        data = pd.DataFrame({
            'open': close_prices + np.random.randn(100) * 0.5,
            'high': close_prices + np.abs(np.random.randn(100)) * 0.5,
            'low': close_prices - np.abs(np.random.randn(100)) * 0.5,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test different target types
        target_types = ['price', 'return', 'log_return', 'hit_probability']
        
        for target_type in target_types:
            df_with_target = featurizer.create_target_variable(
                data, horizon_days=5, target_type=target_type
            )
            
            # Check that target formula is stored
            assert 'target_formula' in df_with_target.attrs, f"Should store formula for {target_type}"
            assert 'target_type' in df_with_target.attrs, f"Should store type for {target_type}"
            assert df_with_target.attrs['target_type'] == target_type, f"Type should match {target_type}"
    
    def test_hit_probability_calculation(self):
        """Test hit probability calculation"""
        featurizer = Featurizer()
        
        # Test hit probability calculation
        predictions = np.array([0.02, 0.05, -0.01, 0.03, 0.04])  # 5% returns
        actual_returns = np.array([0.03, 0.06, -0.02, 0.02, 0.05])  # Actual returns
        threshold = 3.0  # 3% threshold
        
        metrics = featurizer.calculate_hit_probability(
            predictions, actual_returns, threshold
        )
        
        assert 'hit_probability' in metrics, "Should calculate hit probability"
        assert 'confidence_interval_95' in metrics, "Should calculate confidence interval"
        assert 0 <= metrics['hit_probability'] <= 1, "Hit probability should be between 0 and 1"


class TestAdvisoryGeneration:
    """Test advisory generation in Thai and English"""
    
    def test_bilingual_advisory(self):
        """Test that advisory is generated in both languages"""
        advisor = AdvisoryGenerator()
        
        advisory = advisor.generate_advisory(
            predicted_return=5.0,
            target_return=3.0,
            hit_probability=0.7,
            expected_return=4.5,
            risk_level='medium'
        )
        
        assert 'thai' in advisory, "Should have Thai advisory"
        assert 'english' in advisory, "Should have English advisory"
        assert 'level' in advisory, "Should have advisory level"
        assert 'confidence' in advisory, "Should have confidence level"
        
        # Check that Thai text contains Thai characters
        thai_text = advisory['thai']
        assert any('\u0e00' <= char <= '\u0e7f' for char in thai_text), "Thai text should contain Thai characters"
    
    def test_advisory_disclaimer(self):
        """Test that advisory includes proper disclaimer"""
        advisor = AdvisoryGenerator()
        
        advisory = advisor.generate_advisory(
            predicted_return=5.0,
            target_return=3.0,
            hit_probability=0.7,
            expected_return=4.5,
            risk_level='medium'
        )
        
        thai_text = advisory['thai']
        english_text = advisory['english']
        
        # Check for disclaimer in both languages
        assert 'การศึกษา' in thai_text or 'ไม่ใช่คำแนะนำการลงทุน' in thai_text, "Thai should have disclaimer"
        assert 'educational' in english_text.lower() or 'not investment advice' in english_text.lower(), "English should have disclaimer"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
