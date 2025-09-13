# AI Quant Stock Predictor 📈

A comprehensive AI-powered stock prediction system using LSTM and Transformer models for time series forecasting, risk assessment, and backtesting.

## Features

### 🤖 Dual Model Architecture
- **LSTM Model**: Long Short-Term Memory networks for sequential pattern recognition
- **Transformer Model**: Attention-based architecture for complex time series relationships
- **Automatic Model Selection**: Based on validation performance metrics

### 🔧 Advanced Feature Engineering
- **50+ Technical Indicators**: RSI, MACD, ATR, Bollinger Bands, Stochastic, Williams %R, CCI, ROC, MFI, OBV
- **Price Patterns**: Higher highs, lower lows, gap analysis
- **Rolling Statistics**: Multiple time windows (5, 10, 20, 50 days)
- **Volume Analysis**: Volume ratios, changes, and momentum

### 📊 Comprehensive Risk Assessment
- **Prediction Intervals**: 95% confidence ranges
- **Hit Probability**: Likelihood of achieving target returns
- **Expected Return**: Risk-adjusted return calculations
- **Volatility Analysis**: Rolling volatility and risk metrics

### 🎯 Multiple Trading Strategies
- **Scheme A**: Long when predicted return ≥ 0%
- **Scheme B**: Long when predicted return ≥ target return
- **Scheme C**: Long when hit probability ≥ 60%

### 📈 Advanced Backtesting
- **Realistic Costs**: Transaction fees (0.15%) and slippage (0.10%)
- **Walk-Forward Analysis**: Rolling training and testing periods
- **Performance Metrics**: CAGR, Sharpe ratio, Max Drawdown, Win Rate, Profit Factor
- **Risk Metrics**: Volatility, Value at Risk, Expected Shortfall

### 🌐 Bilingual Support
- **Thai & English**: Complete advisory system in both languages
- **Cultural Context**: Localized investment guidance
- **Educational Focus**: Clear disclaimers and educational purpose

### 🎨 Modern UI Design
- **Minimal White Theme**: Clean, professional interface
- **Responsive Design**: Works on desktop and mobile
- **Interactive Elements**: Hover effects and smooth transitions
- **Custom Typography**: Inter font for better readability

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-quant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## Usage

### Quick Start
1. Open the web application in your browser
2. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
3. Select prediction horizon (1, 5, 7, 14, 30 days)
4. Set target return percentage
5. Choose backtesting scheme
6. Click "Train & Compare Models"

### Input Parameters

#### Stock Symbol
- Any valid stock symbol supported by Yahoo Finance
- Examples: AAPL, GOOGL, MSFT, TSLA, AMZN

#### Prediction Horizon
- **1 day**: Short-term intraday predictions
- **5 days**: Weekly trading strategies
- **7 days**: One-week holding periods
- **14 days**: Bi-weekly analysis
- **30 days**: Monthly investment horizons
- **Custom**: Any number of days (1-365)

#### Target Return
- **3%**: Conservative targets
- **5%**: Moderate risk-return
- **10%**: Aggressive growth targets
- **Custom**: Any percentage (0.1-100%)

### Model Selection Criteria

The system automatically selects the best model using a hierarchical approach:

1. **Primary**: Lowest RMSE on validation set
2. **Secondary**: If RMSE difference < 2%, use MAE
3. **Tertiary**: If still tied, use MAPE

### Output Interpretation

#### Model Performance
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **R²**: Coefficient of determination (higher is better)

#### Prediction Results
- **Predicted Price**: Forecasted stock price at horizon
- **Predicted Return**: Expected return percentage
- **Hit Probability**: Likelihood of achieving target (0-100%)
- **Expected Return**: Risk-adjusted expected return

#### Backtest Metrics
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss ratio

## Technical Architecture

### Data Pipeline
1. **Data Fetching**: Yahoo Finance API for OHLCV data
2. **Data Validation**: Quality checks and missing data handling
3. **Feature Engineering**: Technical indicators and pattern recognition
4. **Supervised Windowing**: Time series to supervised learning conversion

### Model Training
1. **Data Splitting**: 70% train, 15% validation, 15% test
2. **Feature Scaling**: StandardScaler fit only on training data
3. **Model Training**: LSTM and Transformer with early stopping
4. **Hyperparameter Tuning**: Optimized default configurations

### Risk Management
1. **Prediction Intervals**: Based on residual analysis
2. **Monte Carlo Simulation**: For uncertainty quantification
3. **Bootstrap Methods**: For robust probability estimates
4. **Stress Testing**: Extreme scenario analysis

### Backtesting Engine
1. **Signal Generation**: Multiple trading schemes
2. **Cost Modeling**: Realistic transaction costs
3. **Walk-Forward Analysis**: Out-of-sample testing
4. **Performance Attribution**: Detailed trade analysis

## Configuration

### Model Parameters
```python
LSTM_CONFIG = {
    'layers': 2,
    'hidden_units': 64,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'epochs': 30,
    'patience': 5,
    'batch_size': 64
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
    'batch_size': 64
}
```

### Backtest Settings
```python
BACKTEST_CONFIG = {
    'fee_bp': 15,  # 0.15% transaction fee
    'slippage_bp': 10,  # 0.10% slippage
    'holding_rule': 'hold_to_horizon',
    'walk_forward_train_months': 36,
    'walk_forward_test_months': 6
}
```

## File Structure

```
ai-quant/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration parameters
├── data_loader.py        # Data fetching and validation
├── featurizer.py         # Feature engineering
├── models.py             # LSTM and Transformer models
├── predictor.py          # Prediction and risk calculation
├── backtester.py         # Backtesting engine
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dependencies

- **Streamlit**: Web application framework
- **TensorFlow**: Deep learning models
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Plotly**: Interactive visualizations
- **YFinance**: Stock data fetching
- **TA**: Technical analysis indicators

## Limitations & Disclaimers

### Data Limitations
- Historical data may not reflect future performance
- Corporate actions may affect price continuity
- Market conditions change over time

### Model Limitations
- Models are trained on historical data only
- No guarantee of future performance
- Black swan events not captured

### Risk Warnings
- **High Risk**: Stock prediction involves significant risk
- **No Guarantees**: Past performance doesn't predict future results
- **Educational Purpose**: Not intended as investment advice
- **Professional Advice**: Consult financial advisors for investment decisions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes only. Please ensure compliance with local financial regulations before using for actual trading.

## Support

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**⚠️ Important Disclaimer**: This software is for educational and research purposes only. It does not constitute financial advice, investment recommendations, or trading signals. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results, and all investments carry risk of loss.
