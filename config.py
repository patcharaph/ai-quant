"""
Configuration file for AI Quant Stock Prediction System
"""

# Model Configuration
LSTM_CONFIG = {
    'layers': 2,
    'hidden_units': 64,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'epochs': 30,
    'patience': 5,
    'batch_size': 64,
    'optimizer': 'adam'
}

TRANSFORMER_CONFIG = {
    'd_model': 64,
    'n_heads': 4,
    'num_layers': 2,
    'ff_dim': 128,
    'dropout': 0.1,
    'learning_rate': 1e-3,
    'epochs': 30,
    'patience': 5,
    'batch_size': 64,
    'optimizer': 'adam'
}

# Data Configuration
DATA_CONFIG = {
    'lookback_window': 60,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'max_years_back': 20,
    'min_data_years': 3
}

# Feature Configuration
FEATURE_CONFIG = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr_period': 14,
    'rolling_windows': [5, 10, 20, 50]
}

# Backtest Configuration
BACKTEST_CONFIG = {
    'fee_bp': 15,  # 0.15%
    'slippage_bp': 10,  # 0.10%
    'holding_rule': 'hold_to_horizon',
    'walk_forward_train_months': 36,
    'walk_forward_test_months': 6,
    'walk_forward_step_months': 6
}

# UI Configuration
UI_CONFIG = {
    'default_horizons': [1, 5, 7, 14, 30],
    'default_targets': [3, 5, 10],
    'chart_height': 400,
    'max_trades_display': 10
}

# Risk Configuration
RISK_CONFIG = {
    'confidence_level': 0.95,
    'z_score': 1.96,  # 95% confidence
    'min_hit_probability': 0.6,
    'uncertain_threshold': 0.4
}

# Advisory Rules
ADVISORY_RULES = {
    'high_confidence': {'min_prob': 0.6, 'min_er_ratio': 0.8},
    'uncertain': {'min_prob': 0.4, 'max_prob': 0.6},
    'low_confidence': {'max_prob': 0.4}
}
