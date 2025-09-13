# 🚀 Quick Start Guide - AI Quant Stock Predictor

## 📋 Prerequisites
- Python 3.8 or higher
- Internet connection for data fetching
- (Optional) OpenRouter API key for LLM advisor

## ⚡ Quick Setup

### Windows Setup (Recommended)
```powershell
# 1. Create virtual environment
python -m venv ai-quant-env
ai-quant-env\Scripts\activate

# 2. Install dependencies (CPU-only, pinned versions)
pip install -r requirements.txt

# 3. Setup environment (optional)
copy env_example.txt .env
# Edit .env and add your OpenRouter API key

# 4. Run the application
streamlit run app.py
```

### Linux/macOS Setup
```bash
# 1. Create virtual environment
python3 -m venv ai-quant-env
source ai-quant-env/bin/activate

# 2. Install dependencies (CPU-only, pinned versions)
pip install -r requirements.txt

# 3. Setup environment (optional)
cp env_example.txt .env
# Edit .env and add your OpenRouter API key

# 4. Run the application
streamlit run app.py
```

### Option 3: Automated Setup
```bash
python setup.py
```

## 🎯 First Steps

1. **Open the app** in your browser (usually http://localhost:8501)

2. **Enter a stock symbol** (e.g., AAPL, GOOGL, MSFT, TSLA)

3. **Set your parameters:**
   - Prediction Horizon: 5 days (recommended)
   - Target Return: 5% (moderate)
   - Backtest Scheme: A (simple)

4. **Enable LLM Advisor** (if you have API key)

5. **Click "Train & Compare Models"**

## 🔧 Configuration

### Environment Variables (.env file)
```bash
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/auto
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Available Models (choose one):
# openrouter/auto - Auto-select best model (recommended)
# openai/gpt-4o-mini - GPT-4o Mini (fast & cheap)
# openai/gpt-3.5-turbo - GPT-3.5 Turbo (balanced)
# anthropic/claude-3-haiku - Claude 3 Haiku (fast)
# anthropic/claude-3-sonnet - Claude 3 Sonnet (high quality)
# google/gemini-pro - Gemini Pro (Google's model)
# meta-llama/llama-3.1-8b-instruct - Llama 3.1 8B (open source)
# mistralai/mixtral-8x7b-instruct - Mixtral 8x7B (efficient)
# qwen/qwen-2.5-7b-instruct - Qwen 2.5 7B (multilingual)

# Optional settings
MAX_TOKENS=500
TEMPERATURE=0.7
```

### Popular Stock Symbols
- **AAPL** - Apple Inc.
- **GOOGL** - Alphabet Inc.
- **MSFT** - Microsoft Corporation
- **TSLA** - Tesla Inc.
- **AMZN** - Amazon.com Inc.
- **META** - Meta Platforms Inc.

## 📊 Understanding the Results

### Model Performance
- **RMSE**: Lower is better (Root Mean Square Error)
- **MAE**: Lower is better (Mean Absolute Error)
- **R²**: Higher is better (Coefficient of determination)

### Prediction Results
- **Predicted Price**: AI forecast for target date
- **Predicted Return**: Expected percentage return
- **Hit Probability**: Likelihood of achieving target
- **Expected Return**: Risk-adjusted expected return

### Backtest Metrics
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## 🎨 Features Overview

### 🤖 AI Models
- **LSTM**: Long Short-Term Memory for sequential patterns
- **Transformer**: Attention-based architecture
- **Auto Selection**: Best model chosen automatically

### 📈 Analytics
- **50+ Indicators**: RSI, MACD, ATR, Bollinger Bands, etc.
- **Pattern Recognition**: Price patterns and volume analysis
- **Risk Assessment**: Prediction intervals and probabilities

### 🌐 Advisory System
- **Bilingual**: Thai and English support
- **LLM Enhanced**: AI-powered human-readable advice
- **Educational Focus**: Clear disclaimers and guidance

## 🚨 Troubleshooting

### Common Issues

**1. Data Loading Error**
```
Error: No data found for symbol
```
- Try different stock symbol
- Check internet connection
- Some symbols may not be available

**2. LLM Not Working**
```
LLM not configured
```
- Check .env file exists
- Verify API key is correct
- Ensure OpenRouter account is active

**3. Memory Issues**
```
Out of memory
```
- Reduce lookback window in config
- Use smaller batch size
- Close other applications

**4. Import Errors**
```
ModuleNotFoundError
```
- Run: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Use virtual environment

### Performance Tips

1. **Start with popular symbols** (AAPL, GOOGL, MSFT)
2. **Use shorter horizons** (1-7 days) for faster training
3. **Enable LLM only when needed** to save API costs
4. **Close other applications** to free up memory

## 📚 Additional Resources

- **README.md**: Comprehensive documentation
- **SYSTEM_SUMMARY.md**: Technical details
- **demo.py**: Example with real data
- **test_mock.py**: Test with mock data

## 🆘 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review the README.md
3. Try the mock data example
4. Check your internet connection
5. Verify all dependencies are installed

## ⚠️ Important Notes

- **Educational Purpose**: This tool is for learning and research
- **Not Investment Advice**: Always consult financial professionals
- **Risk Warning**: All investments carry risk of loss
- **Data Accuracy**: Historical data may not reflect future performance

---

**Happy Trading! 📈🚀**
