"""
Test the AI Quant system with mock data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from featurizer import Featurizer
from models import LSTMModel, TransformerModel, ModelEvaluator, ModelSelector
from predictor import Predictor, AdvisoryGenerator
import config

def create_mock_data(n_days=1000):
    """Create mock OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data (random walk with trend)
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily return, 2% volatility
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    
    # Generate open, high, low based on close
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(dates)))
    data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
    data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
    data['volume'] = np.random.randint(1000000, 10000000, len(dates))
    
    # Remove NaN values
    data = data.dropna()
    
    return data

def test_system():
    """Test the complete system with mock data"""
    print("🚀 AI Quant Stock Predictor - Mock Data Test")
    print("=" * 60)
    
    # Create mock data
    print("📊 Creating mock data...")
    data = create_mock_data(1000)
    print(f"✅ Created {len(data)} days of mock data")
    
    # Initialize components
    featurizer = Featurizer(config.FEATURE_CONFIG)
    lstm_model = LSTMModel(config.LSTM_CONFIG)
    transformer_model = TransformerModel(config.TRANSFORMER_CONFIG)
    predictor = Predictor(config.RISK_CONFIG)
    advisory_generator = AdvisoryGenerator(config.ADVISORY_RULES)
    
    # Test parameters
    horizon_days = 5
    target_return_pct = 3.0
    
    print(f"🎯 Horizon: {horizon_days} days")
    print(f"💰 Target Return: {target_return_pct}%")
    print()
    
    try:
        # Step 1: Feature engineering
        print("🔧 Engineering features...")
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = featurizer.make_supervised(
            data, config.DATA_CONFIG['lookback_window'], horizon_days, 'price'
        )
        print(f"✅ Created {metadata['n_features']} features from {metadata['n_samples']} samples")
        print(f"   - Training: {metadata['n_train']} samples")
        print(f"   - Validation: {metadata['n_val']} samples")
        print(f"   - Test: {metadata['n_test']} samples")
        
        # Step 2: Train models
        print("\n🤖 Training LSTM model...")
        lstm_model, lstm_history = lstm_model.train(X_train, y_train, X_val, y_val)
        print("✅ LSTM training completed")
        
        print("🤖 Training Transformer model...")
        transformer_model, trans_history = transformer_model.train(X_train, y_train, X_val, y_val)
        print("✅ Transformer training completed")
        
        # Step 3: Evaluate models
        print("\n📊 Evaluating models...")
        
        # LSTM evaluation
        lstm_val_pred = lstm_model.predict(X_val)
        lstm_test_pred = lstm_model.predict(X_test)
        lstm_val_metrics = ModelEvaluator.calculate_metrics(y_val, lstm_val_pred)
        lstm_test_metrics = ModelEvaluator.calculate_metrics(y_test, lstm_test_pred)
        
        # Transformer evaluation
        trans_val_pred = transformer_model.predict(X_val)
        trans_test_pred = transformer_model.predict(X_test)
        trans_val_metrics = ModelEvaluator.calculate_metrics(y_val, trans_val_pred)
        trans_test_metrics = ModelEvaluator.calculate_metrics(y_test, trans_test_pred)
        
        # Step 4: Model selection
        print("\n🏆 Selecting best model...")
        winner, reason = ModelSelector.select_best_model(lstm_val_metrics, trans_val_metrics)
        print(f"✅ Best Model: {winner.upper()}")
        print(f"📝 Reason: {reason}")
        
        # Step 5: Generate prediction
        print("\n🔮 Generating prediction...")
        
        if winner == 'lstm':
            best_model = lstm_model
            best_pred = lstm_test_pred
            best_metrics = lstm_test_metrics
        else:
            best_model = transformer_model
            best_pred = trans_test_pred
            best_metrics = trans_test_metrics
        
        # Make forecast
        latest_sequence = X_test[-1:].reshape(1, -1, X_test.shape[-1])
        forecast = predictor.forecast(best_model, latest_sequence[0], horizon_days)
        predicted_price = float(forecast['y_hat'])
        current_price = float(data['close'].iloc[-1])
        predicted_return = float(predictor.calculate_predicted_return(predicted_price, current_price, 'price'))
        
        # Risk calculations
        residuals = y_test - best_pred
        pi_stats = predictor.calculate_prediction_interval(residuals)
        hit_probability = float(predictor.calculate_hit_probability(
            predicted_return, target_return_pct, pi_stats['residual_std']
        ))
        expected_return = predictor.calculate_expected_return(
            predicted_return, hit_probability, target_return_pct
        )
        
        # Generate advisory
        advisory = advisory_generator.generate_advisory(
            float(predicted_return), target_return_pct, float(hit_probability),
            float(expected_return['expected_return'])
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("📈 PREDICTION RESULTS")
        print("=" * 60)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${predicted_price:.2f}")
        print(f"Predicted Return: {predicted_return:.2f}%")
        print(f"Hit Probability: {hit_probability:.1%}")
        print(f"Expected Return: {expected_return['expected_return']:.2f}%")
        print(f"Prediction Interval: ±{pi_stats['residual_std']:.2f}")
        
        print("\n" + "=" * 60)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("=" * 60)
        
        print("LSTM Validation Metrics:")
        for metric, value in lstm_val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nLSTM Test Metrics:")
        for metric, value in lstm_test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nTransformer Validation Metrics:")
        for metric, value in trans_val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nTransformer Test Metrics:")
        for metric, value in trans_test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n" + "=" * 60)
        print("💡 INVESTMENT ADVISORY")
        print("=" * 60)
        print("🇹🇭 Thai:")
        print(advisory['thai'])
        print("\n🇺🇸 English:")
        print(advisory['english'])
        
        # Test backtesting components
        print("\n" + "=" * 60)
        print("📈 BACKTESTING TEST")
        print("=" * 60)
        
        from backtester import SignalGenerator, Backtester
        
        # Create mock predictions for backtesting
        predictions_df = pd.DataFrame({
            'predicted_return': [predicted_return] * len(data),
            'hit_probability': [hit_probability] * len(data)
        }, index=data.index)
        
        signal_generator = SignalGenerator()
        backtester = Backtester(config.BACKTEST_CONFIG)
        
        # Generate signals
        signals = signal_generator.generate_signals(
            predictions_df, data['close'], 'A', target_return_pct
        )
        
        print(f"✅ Generated signals: {signals['signal'].sum()} buy signals out of {len(signals)} periods")
        
        # Run backtest
        trades_df, equity_df, backtest_metrics = backtester.run_backtest(
            data['close'], signals, horizon_days
        )
        
        print(f"✅ Backtest completed: {backtest_metrics['N_Trades']} trades")
        print(f"   - CAGR: {backtest_metrics['CAGR']:.2f}%")
        print(f"   - Sharpe: {backtest_metrics['Sharpe']:.2f}")
        print(f"   - Max Drawdown: {backtest_metrics['Max_Drawdown']:.2f}%")
        print(f"   - Win Rate: {backtest_metrics['Win_Rate']:.1f}%")
        
        print("\n✅ All tests completed successfully!")
        print("🎉 The AI Quant Stock Predictor system is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()
