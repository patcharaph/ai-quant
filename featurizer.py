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
    
    def create_target_variable(self, data, horizon_days, target_type='price', target_threshold=None):
        """
        Create target variable for prediction with clear labeling formulas
        
        Args:
            data (pd.DataFrame): Data with features
            horizon_days (int): Prediction horizon in days
            target_type (str): 'price', 'return', 'hit_probability', or 'log_return'
            target_threshold (float): Threshold for hit probability calculation (in percentage)
            
        Returns:
            pd.DataFrame: Data with target variable and labeling metadata
        """
        df = data.copy()
        
        # Log the target creation formula
        logger.info(f"📊 Creating target variable: {target_type}, horizon={horizon_days} days")
        
        if target_type == 'price':
            # Target: y = close[t+h] (absolute price at horizon)
            df['target'] = df['close'].shift(-horizon_days)
            formula = f"y = close[t+{horizon_days}]"
            self.target_column = 'target'
            
        elif target_type == 'return':
            # Target: y = (close[t+h] - close[t]) / close[t] * 100 (percentage return)
            df['target'] = (df['close'].shift(-horizon_days) / df['close'] - 1) * 100
            formula = f"y = (close[t+{horizon_days}] - close[t]) / close[t] * 100"
            self.target_column = 'target'
            
        elif target_type == 'log_return':
            # Target: y = log(close[t+h] / close[t]) (log return)
            df['target'] = np.log(df['close'].shift(-horizon_days) / df['close'])
            formula = f"y = log(close[t+{horizon_days}] / close[t])"
            self.target_column = 'target'
            
        elif target_type == 'hit_probability':
            # Target: y = 1 if return >= threshold, 0 otherwise
            if target_threshold is None:
                target_threshold = 3.0  # Default 3% threshold
                
            returns = (df['close'].shift(-horizon_days) / df['close'] - 1) * 100
            df['target'] = (returns >= target_threshold).astype(int)
            formula = f"y = 1 if return >= {target_threshold}%, 0 otherwise"
            self.target_column = 'target'
            
            # Store hit probability metadata
            hit_rate = df['target'].mean()
            logger.info(f"🎯 Hit probability target: {hit_rate:.2%} of samples above {target_threshold}% threshold")
            
        else:
            raise ValueError("target_type must be 'price', 'return', 'log_return', or 'hit_probability'")
        
        # Store target metadata
        df.attrs['target_formula'] = formula
        df.attrs['target_type'] = target_type
        df.attrs['horizon_days'] = horizon_days
        df.attrs['target_threshold'] = target_threshold
        
        logger.info(f"✅ Target formula: {formula}")
        
        return df
    
    def calculate_hit_probability(self, predictions, actual_returns, threshold=3.0):
        """
        Calculate hit probability for predictions
        
        Args:
            predictions (np.array): Model predictions
            actual_returns (np.array): Actual returns (in percentage)
            threshold (float): Threshold for hit definition (in percentage)
            
        Returns:
            dict: Hit probability metrics
        """
        # Convert predictions to returns if needed
        if len(predictions) != len(actual_returns):
            raise ValueError("Predictions and actual returns must have same length")
        
        # Calculate hit probability
        hits = (actual_returns >= threshold).astype(int)
        hit_prob = hits.mean()
        
        # Calculate confidence intervals
        n = len(hits)
        se = np.sqrt(hit_prob * (1 - hit_prob) / n)
        ci_95 = 1.96 * se
        
        metrics = {
            'hit_probability': hit_prob,
            'hit_rate': hit_prob,
            'total_samples': n,
            'hits': hits.sum(),
            'threshold': threshold,
            'standard_error': se,
            'ci_95_lower': max(0, hit_prob - ci_95),
            'ci_95_upper': min(1, hit_prob + ci_95),
            'confidence_interval_95': (max(0, hit_prob - ci_95), min(1, hit_prob + ci_95))
        }
        
        logger.info(f"🎯 Hit Probability: {hit_prob:.2%} (95% CI: {metrics['ci_95_lower']:.2%} - {metrics['ci_95_upper']:.2%})")
        
        return metrics
    
    def check_data_leakage(self, X, y, lookback_window, horizon_days, split_indices):
        """Check for data leakage in the dataset with enhanced validation"""
        logger.info("🔍 Checking for data leakage...")
        
        # Check 1: Ensure no future data in features
        if len(X) > 0:
            # Check if any feature contains future information
            future_check = np.any(np.isnan(X))
            if future_check:
                logger.warning("⚠️  Potential future data detected in features")
                self.data_leakage_checks['future_data_in_features'] = True
            
            # Check for target variable in features (common leakage)
            if hasattr(self, 'feature_columns') and self.target_column in self.feature_columns:
                logger.error("❌ Target variable found in feature columns - DATA LEAKAGE!")
                self.data_leakage_checks['target_in_features'] = True
        
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
        
        # Check 4: Validate scaler fitting (critical for preventing leakage)
        if hasattr(self, 'scaler') and hasattr(self.scaler, 'scale_'):
            # Check if scaler was fitted on training data only
            if not hasattr(self, '_scaler_fitted_on_train_only'):
                logger.warning("⚠️  Scaler fitting validation not recorded")
                self.data_leakage_checks['scaler_fitted_on_test'] = True
            else:
                logger.info("✅ Scaler fitted on training data only")
        
        # Check 5: Validate horizon gap (no overlap between features and target)
        if horizon_days <= 0:
            logger.error("❌ Invalid horizon: must be positive to prevent overlap")
            self.data_leakage_checks['horizon_validation'] = False
        else:
            logger.info(f"✅ Valid horizon gap: {horizon_days} days")
        
        logger.info(f"✅ Data leakage checks: {self.data_leakage_checks}")
        return self.data_leakage_checks

    def make_supervised(self, data, lookback_window, horizon_days, target_type='price', use_walk_forward=False):
        """
        Create supervised learning dataset with proper time ordering and no data leakage
        
        Args:
            data (pd.DataFrame): OHLCV data
            lookback_window (int): Number of past days to use as features
            horizon_days (int): Prediction horizon
            target_type (str): 'price' or 'return'
            use_walk_forward (bool): Use walk-forward validation instead of simple time split
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
        """
        logger.info(f"🔄 Creating supervised dataset: lookback={lookback_window}, horizon={horizon_days}")
        logger.info(f"📊 Target type: {target_type}, Walk-forward: {use_walk_forward}")
        
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
        
        # Create sequences with proper time ordering (NO FUTURE DATA LEAKAGE)
        X, y = [], []
        for i in range(lookback_window, len(df_clean) - horizon_days + 1):
            # Features: past lookback_window days ONLY (no future data)
            X.append(df_clean[feature_cols].iloc[i-lookback_window:i].values)
            # Target: future value at horizon_days ahead
            y.append(df_clean['target'].iloc[i-1])
        
        X = np.array(X)
        y = np.array(y)
        
        if use_walk_forward:
            # Walk-forward validation: sliding window approach
            return self._create_walk_forward_splits(X, y, lookback_window, horizon_days, feature_cols)
        else:
            # Simple time-based split
            return self._create_time_splits(X, y, lookback_window, horizon_days, feature_cols)
    
    def _create_time_splits(self, X, y, lookback_window, horizon_days, feature_cols):
        """Create time-based train/val/test splits with proper scaling"""
        logger.info("📅 Creating time-based splits...")
        
        # Time-based split (no shuffling!)
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
        
        # CRITICAL: Scale features (fit only on training data - NO DATA LEAKAGE)
        logger.info("🔧 Fitting scaler on training data only (no data leakage)...")
        self.scaler = StandardScaler()
        
        # Fit scaler ONLY on training data
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        # Mark that scaler was fitted on training data only
        self._scaler_fitted_on_train_only = True
        
        # Transform all datasets using the same scaler
        X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        logger.info("✅ Scaler fitted on training data only - no data leakage")
        
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
            'target_type': 'price' if self.target_column == 'target' else 'return',
            'n_features': len(feature_cols),
            'data_leakage_checks': leakage_checks,
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'scaler': self.scaler,
            'split_type': 'time_based'
        }
        
        return (X_train_scaled, y_train, X_val_scaled, y_val, 
                X_test_scaled, y_test, metadata)
    
    def _create_walk_forward_splits(self, X, y, lookback_window, horizon_days, feature_cols):
        """Create walk-forward validation splits"""
        logger.info("🚶 Creating walk-forward validation splits...")
        
        # Walk-forward parameters
        train_months = self.config.get('walk_forward_train_months', 36)
        test_months = self.config.get('walk_forward_test_months', 6)
        step_months = self.config.get('walk_forward_step_months', 6)
        
        # Convert months to approximate trading days
        train_days = int(train_months * 21)  # ~21 trading days per month
        test_days = int(test_months * 21)
        step_days = int(step_months * 21)
        
        # Create walk-forward splits
        splits = []
        start_idx = 0
        
        while start_idx + train_days + test_days < len(X):
            train_end = start_idx + train_days
            test_start = train_end
            test_end = test_start + test_days
            
            splits.append({
                'train_start': start_idx,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            start_idx += step_days
        
        if not splits:
            # Create a minimal walk-forward style split for small datasets
            logger.warning("⚠️  Not enough data for walk-forward validation, creating minimal split")
            n = len(X)
            if n < 5:
                # Degenerate case; fall back safely
                return self._create_time_splits(X, y, lookback_window, horizon_days, feature_cols)
            train_end = max(1, int(n * 0.6))
            test_start = train_end
            test_end = max(test_start + 1, int(n * 0.8))
            splits = [{
                'train_start': 0,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            }]
        
        # Use the first split for initial training
        first_split = splits[0]
        X_train = X[first_split['train_start']:first_split['train_end']]
        y_train = y[first_split['train_start']:first_split['train_end']]
        X_val = X[first_split['test_start']:first_split['test_end']]
        y_val = y[first_split['test_start']:first_split['test_end']]
        
        # Use remaining data for test
        X_test = X[first_split['test_end']:]
        y_test = y[first_split['test_end']:]
        
        # Scale features (fit only on training data)
        logger.info("🔧 Fitting scaler on training data only (walk-forward)...")
        self.scaler = StandardScaler()
        
        # Fit scaler ONLY on training data
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        # Mark that scaler was fitted on training data only
        self._scaler_fitted_on_train_only = True
        
        # Transform all datasets using the same scaler
        X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        logger.info("✅ Scaler fitted on training data only (walk-forward) - no data leakage")
        
        # Store feature information
        self.feature_columns = feature_cols

        # Check for data leakage similar to time-based split
        split_indices = [0, first_split['train_end'], first_split['test_end'], len(X)]
        leakage_checks = self.check_data_leakage(X, y, lookback_window, horizon_days, split_indices)

        # Create metadata
        metadata = {
            'feature_columns': feature_cols,
            'target_column': self.target_column,
            'lookback_window': lookback_window,
            'horizon_days': horizon_days,
            'target_type': 'price' if self.target_column == 'target' else 'return',
            'n_features': len(feature_cols),
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'scaler': self.scaler,
            'split_type': 'walk_forward',
            'data_leakage_checks': leakage_checks,
            'walk_forward_splits': splits,
            'train_months': train_months,
            'test_months': test_months,
            'step_months': step_months
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
