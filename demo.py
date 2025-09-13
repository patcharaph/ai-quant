"""
Demo script to test the AI Quant Stock Predictor system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import DataLoader
from featurizer import Featurizer
from models import LSTMModel, TransformerModel, ModelEvaluator, ModelSelector
from predictor import Predictor, AdvisoryGenerator
from backtester import Backtester, SignalGenerator
import config

def run_demo():
    """Run a quick demo of the system"""
    print("🚀 AI Quant Stock Predictor - Demo")
    print("=" * 50)
    
    # Initialize components
    data_loader = DataLoader()
    featurizer = Featurizer(config.FEATURE_CONFIG)
    lstm_model = LSTMModel(config.LSTM_CONFIG)
    transformer_model = TransformerModel(config.TRANSFORMER_CONFIG)
    predictor = Predictor(config.RISK_CONFIG)
    advisory_generator = AdvisoryGenerator(config.ADVISORY_RULES)
    
    # Demo parameters
    symbol = "MSFT"  # Try Microsoft instead
    horizon_days = 5
    target_return_pct = 5.0
    
    print(f"📊 Analyzing: {symbol}")
    print(f"🎯 Horizon: {horizon_days} days")
    print(f"💰 Target Return: {target_return_pct}%")
    print()
    
    try:
        # Step 1: Load data
        print("📥 Loading data...")
        try:
            data = data_loader.fetch_ohlcv(symbol)
        except Exception as e:
            print(f"❌ Failed to fetch {symbol}: {e}")
            print("🔄 Trying alternative symbol...")
            symbol = "GOOGL"  # Try Google
            data = data_loader.fetch_ohlcv(symbol)
        
        data = data_loader.clean_data(data)
        print(f"✅ Loaded {len(data)} days of data for {symbol}")
        
        # Step 2: Feature engineering
        print("🔧 Engineering features...")
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = featurizer.make_supervised(
            data, config.DATA_CONFIG['lookback_window'], horizon_days, 'price'
        )
        print(f"✅ Created {metadata['n_features']} features from {metadata['n_samples']} samples")
        
        # Step 3: Train models (simplified for demo)
        print("🤖 Training models...")
        
        # LSTM
        lstm_model, lstm_history = lstm_model.train(X_train, y_train, X_val, y_val)
        lstm_val_pred = lstm_model.predict(X_val)
        lstm_test_pred = lstm_model.predict(X_test)
        
        # Transformer
        transformer_model, trans_history = transformer_model.train(X_train, y_train, X_val, y_val)
        trans_val_pred = transformer_model.predict(X_val)
        trans_test_pred = transformer_model.predict(X_test)
        
        print("✅ Models trained successfully")
        
        # Step 4: Evaluate models
        print("📊 Evaluating models...")
        
        lstm_val_metrics = ModelEvaluator.calculate_metrics(y_val, lstm_val_pred)
        lstm_test_metrics = ModelEvaluator.calculate_metrics(y_test, lstm_test_pred)
        trans_val_metrics = ModelEvaluator.calculate_metrics(y_val, trans_val_pred)
        trans_test_metrics = ModelEvaluator.calculate_metrics(y_test, trans_test_pred)
        
        # Step 5: Model selection
        winner, reason = ModelSelector.select_best_model(lstm_val_metrics, trans_val_metrics)
        print(f"🏆 Best Model: {winner.upper()}")
        print(f"📝 Reason: {reason}")
        
        # Step 6: Generate prediction
        print("🔮 Generating prediction...")
        
        if winner == 'lstm':
            best_model = lstm_model
            best_pred = lstm_test_pred
        else:
            best_model = transformer_model
            best_pred = trans_test_pred
        
        # Make forecast
        latest_sequence = X_test[-1:].reshape(1, -1, X_test.shape[-1])
        forecast = predictor.forecast(best_model, latest_sequence[0], horizon_days)
        predicted_price = forecast['y_hat']
        current_price = data['close'].iloc[-1]
        predicted_return = predictor.calculate_predicted_return(predicted_price, current_price, 'price')
        
        # Risk calculations
        residuals = y_test - best_pred
        pi_stats = predictor.calculate_prediction_interval(residuals)
        hit_probability = predictor.calculate_hit_probability(
            predicted_return, target_return_pct, pi_stats['residual_std']
        )
        expected_return = predictor.calculate_expected_return(
            predicted_return, hit_probability, target_return_pct
        )
        
        # Generate advisory
        advisory = advisory_generator.generate_advisory(
            predicted_return, target_return_pct, hit_probability,
            expected_return['expected_return']
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("📈 PREDICTION RESULTS")
        print("=" * 50)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${predicted_price:.2f}")
        print(f"Predicted Return: {predicted_return:.2f}%")
        print(f"Hit Probability: {hit_probability:.1%}")
        print(f"Expected Return: {expected_return['expected_return']:.2f}%")
        
        print("\n" + "=" * 50)
        print("📊 MODEL PERFORMANCE")
        print("=" * 50)
        print("LSTM Validation Metrics:")
        for metric, value in lstm_val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nTransformer Validation Metrics:")
        for metric, value in trans_val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n" + "=" * 50)
        print("💡 INVESTMENT ADVISORY")
        print("=" * 50)
        print("🇹🇭 Thai:")
        print(advisory['thai'])
        print("\n🇺🇸 English:")
        print(advisory['english'])
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
