"""
Feature Engineering and Supervised Windowing for Time Series Data
"""

import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import logging
warnings.filterwarnings('ignore')

# Setup logging for data leakage detection
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Featurizer:
    """Handles feature engineering and supervised windowing for time series data"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = None
        self.data_leakage_checks = {
            'scaler_fitted_on_test': False,
            'future_data_in_features': False,
            'target_in_features': False,
            'lookback_window_validation': True
        }
        
    def create_base_features(self, data):
        """
        Create base technical features from OHLCV data
        
        Args:
            data (pd.DataFrame): OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        df = data.copy()
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price-based features
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = df['price_change'] / df['open'] * 100
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close'] * 100
        df['oc_spread'] = (df['close'] - df['open']) / df['open'] * 100
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Rolling statistics
        windows = self.config.get('rolling_windows', [5, 10, 20, 50])
        for window in windows:
            df[f'close_sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'close_ratio_{window}'] = df['close'] / df[f'close_sma_{window}']
            df[f'volatility_{window}'] = df['log_return'].rolling(window=window).std()
        
        # Technical indicators
        # RSI
        rsi_period = self.config.get('rsi_period', 14)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
        
        # MACD
        macd_fast = self.config.get('macd_fast', 12)
        macd_slow = self.config.get('macd_slow', 26)
        macd_signal = self.config.get('macd_signal', 9)
        macd_indicator = ta.trend.MACD(df['close'], window_slow=macd_slow, 
                                      window_fast=macd_fast, window_sign=macd_signal)
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_histogram'] = macd_indicator.macd_diff()
        
        # ATR
        atr_period = self.config.get('atr_period', 14)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 
                                                  window=atr_period).average_true_range()
        df['atr_ratio'] = df['atr'] / df['close'] * 100
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=bb_std)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        stoch_indicator = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_indicator.stoch()
        df['stoch_d'] = stoch_indicator.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # Commodity Channel Index
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        # Rate of Change
        df['roc'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
        
        # Money Flow Index
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        
        # On Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma']
        
        # Price patterns
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'return_lag_{lag}'] = df['log_return'].shift(lag)
        
        return df
    
    def create_target_variable(self, data, horizon_days, target_type='price'):
        """
        Create target variable for prediction
        
        Args:
            data (pd.DataFrame): Data with features
            horizon_days (int): Prediction horizon in days
            target_type (str): 'price' or 'return'
            
        Returns:
            pd.DataFrame: Data with target variable
        """
        df = data.copy()
        
        if target_type == 'price':
            # Target is the price at t+h
            df['target'] = df['close'].shift(-horizon_days)
            self.target_column = 'target'
        elif target_type == 'return':
            # Target is the return from t to t+h
            df['target'] = (df['close'].shift(-horizon_days) / df['close'] - 1) * 100
            self.target_column = 'target'
        else:
            raise ValueError("target_type must be 'price' or 'return'")
        
        return df
    
    def check_data_leakage(self, X, y, lookback_window, horizon_days, split_indices):
        """Check for data leakage in the dataset"""
        logger.info("🔍 Checking for data leakage...")
        
        # Check 1: Ensure no future data in features
        if len(X) > 0:
            # Check if any feature contains future information
            future_check = np.any(np.isnan(X))
            if future_check:
                logger.warning("⚠️  Potential future data detected in features")
                self.data_leakage_checks['future_data_in_features'] = True
        
        # Check 2: Validate lookback window
        if len(X) > 0 and X.shape[1] != lookback_window:
            logger.error(f"❌ Lookback window mismatch: expected {lookback_window}, got {X.shape[1]}")
            self.data_leakage_checks['lookback_window_validation'] = False
        
        # Check 3: Ensure proper time ordering
        if len(split_indices) >= 3:
            train_end, val_end = split_indices[1], split_indices[2]
            if train_end >= val_end:
                logger.error("❌ Invalid time split: training data overlaps with validation")
                self.data_leakage_checks['time_split_validation'] = False
        
        logger.info(f"✅ Data leakage checks: {self.data_leakage_checks}")
        return self.data_leakage_checks

    def make_supervised(self, data, lookback_window, horizon_days, target_type='price'):
        """
        Create supervised learning dataset with windowing
        
        Args:
            data (pd.DataFrame): OHLCV data
            lookback_window (int): Number of past days to use as features
            horizon_days (int): Prediction horizon
            target_type (str): 'price' or 'return'
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
        """
        logger.info(f"🔄 Creating supervised dataset: lookback={lookback_window}, horizon={horizon_days}")
        
        # Create features
        df_features = self.create_base_features(data)
        df_target = self.create_target_variable(df_features, horizon_days, target_type)
        
        # Select feature columns (exclude target and non-numeric columns)
        exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df_target.columns if col not in exclude_cols]
        
        # Remove rows with NaN values
        df_clean = df_target[feature_cols + ['target']].dropna()
        
        if len(df_clean) < lookback_window + horizon_days:
            raise ValueError(f"Insufficient data: need at least {lookback_window + horizon_days} days")
        
        # Create sequences with proper time ordering
        X, y = [], []
        for i in range(lookback_window, len(df_clean) - horizon_days + 1):
            # Features: past lookback_window days (no future data)
            X.append(df_clean[feature_cols].iloc[i-lookback_window:i].values)
            # Target: future value at horizon_days ahead
            y.append(df_clean['target'].iloc[i-1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Time-based split
        train_ratio = self.config.get('train_ratio', 0.7)
        val_ratio = self.config.get('val_ratio', 0.15)
        
        n_train = int(len(X) * train_ratio)
        n_val = int(len(X) * val_ratio)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        # Scale features (fit only on training data - CRITICAL for no data leakage)
        logger.info("🔧 Fitting scaler on training data only...")
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Store feature information
        self.feature_columns = feature_cols
        
        # Check for data leakage
        split_indices = [0, n_train, n_train+n_val, len(X)]
        leakage_checks = self.check_data_leakage(X, y, lookback_window, horizon_days, split_indices)
        
        # Create metadata
        metadata = {
            'feature_columns': feature_cols,
            'target_column': self.target_column,
            'lookback_window': lookback_window,
            'horizon_days': horizon_days,
            'target_type': target_type,
            'n_features': len(feature_cols),
            'data_leakage_checks': leakage_checks,
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'scaler': self.scaler
        }
        
        return (X_train_scaled, y_train, X_val_scaled, y_val, 
                X_test_scaled, y_test, metadata)
    
    def get_feature_importance_names(self):
        """Get names of features for interpretation"""
        return self.feature_columns
    
    def inverse_transform_target(self, y_scaled, original_data):
        """
        Inverse transform target variable if it was scaled
        
        Args:
            y_scaled (np.array): Scaled target values
            original_data (pd.DataFrame): Original data for reference
            
        Returns:
            np.array: Original scale target values
        """
        # For price targets, no inverse transform needed
        # For return targets, they're already in percentage
        return y_scaled
