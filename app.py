"""
AI Quant Stock Prediction System - Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import DataLoader
from featurizer import Featurizer
from models import LSTMModel, TransformerModel, ModelEvaluator, ModelSelector
from predictor import Predictor, RiskCalculator, AdvisoryGenerator
from backtester import Backtester, SignalGenerator, BacktestAnalyzer
from llm_advisor import LLMAdvisor
import config
import os
from env_manager import get_env_config

# Page configuration
st.set_page_config(
    page_title="AI Quant Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Professional Theme with Thai Support
st.markdown("""
<style>
    /* Import Google Fonts with Thai Support */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        font-family: 'Inter', 'Noto Sans Thai', sans-serif;
        color: #ffffff;
    }
    
    /* Thai Text Support */
    .thai-text {
        font-family: 'Noto Sans Thai', 'Inter', sans-serif;
        font-weight: 400;
    }
    
    .thai-title {
        font-family: 'Noto Sans Thai', 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    .thai-subtitle {
        font-family: 'Noto Sans Thai', 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(45deg, #00d4ff, #00ff88, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%);
        border-right: 2px solid #00d4ff;
        box-shadow: 2px 0 15px rgba(0, 212, 255, 0.3);
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background-color: #2d2d2d;
        border: 2px solid #00d4ff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
        color: #ffffff;
    }
    
    .css-1d391kg .stTextInput > div > div > input {
        background-color: #2d2d2d;
        border: 2px solid #00d4ff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
        color: #ffffff;
        font-weight: 500;
    }
    
    .css-1d391kg .stButton > button {
        background: linear-gradient(45deg, #2d2d2d, #404040);
        color: #ffffff;
        border: 2px solid #00d4ff;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
    }
    
    .css-1d391kg .stButton > button:hover {
        background: linear-gradient(45deg, #404040, #555555);
        border-color: #00ff88;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 255, 136, 0.4);
    }
    
    /* Primary Button */
    .css-1d391kg .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #00d4ff, #00ff88);
        color: #000000;
        border: 2px solid #00d4ff;
        font-weight: 700;
    }
    
    .css-1d391kg .stButton > button[kind="primary"]:hover {
        background: linear-gradient(45deg, #00b8e6, #00e677);
        border-color: #00ff88;
        transform: translateY(-2px);
    }
    
    /* Secondary Button */
    .css-1d391kg .stButton > button[kind="secondary"] {
        background: linear-gradient(45deg, #2d2d2d, #404040);
        color: #ffffff;
        border: 2px solid #ff6b35;
        font-weight: 600;
    }
    
    .css-1d391kg .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(45deg, #404040, #555555);
        border-color: #ff8c42;
        transform: translateY(-2px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #404040 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #00d4ff;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        color: #ffffff;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00d4ff, #00ff88, #ff6b35);
    }
    
    .metric-card:hover {
        box-shadow: 0 12px 35px rgba(0, 212, 255, 0.4);
        transform: translateY(-5px);
        border-color: #00ff88;
    }
    
    /* Advisory Boxes */
    .advisory-box {
        background: linear-gradient(135deg, #2d2d2d 0%, #404040 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #00d4ff;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        position: relative;
        overflow: hidden;
        color: #ffffff;
    }
    
    .advisory-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00d4ff, #00ff88, #ff6b35);
    }
    
    .advisory-box h3 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    .advisory-box p {
        color: #e0e0e0;
        line-height: 1.7;
        margin-bottom: 0.8rem;
        font-weight: 500;
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #fffbf0 100%);
        padding: 1.5rem;
        border-radius: 20px;
        border: 3px solid #ffd93d;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 217, 61, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .warning-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ffd93d, #ffb347);
    }
    
    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, #f1f8e9 0%, #f8fff8 100%);
        padding: 1.5rem;
        border-radius: 20px;
        border: 3px solid #81c784;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(129, 199, 132, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .success-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #81c784, #66bb6a);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%);
        padding: 1.5rem;
        border-radius: 20px;
        border: 3px solid #64b5f6;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(100, 181, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #64b5f6, #42a5f5);
    }
    
    /* Tables */
    .stDataFrame {
        background: linear-gradient(135deg, #2d2d2d 0%, #404040 100%);
        border-radius: 15px;
        border: 2px solid #00d4ff;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        color: #ffffff;
    }
    
    /* Charts */
    .js-plotly-plot {
        background: linear-gradient(135deg, #2d2d2d 0%, #404040 100%);
        border-radius: 15px;
        border: 2px solid #00d4ff;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 212, 255, 0.3);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        border: 2px solid #00d4ff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
        color: #ffffff;
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        border: 2px solid #00d4ff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        background-color: #2d2d2d;
        border: 2px solid #00d4ff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Checkbox */
    .stCheckbox > div > div > div {
        background-color: #2d2d2d;
        border: 2px solid #00d4ff;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 212, 255, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* Markdown Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Links */
    a {
        color: #1a1a1a;
        text-decoration: none;
    }
    
    a:hover {
        color: #333333;
        text-decoration: underline;
    }
    
    /* Code Blocks */
    .stCodeBlock {
        background-color: #f8f9fa;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 12px;
        border: 1px solid #e5e5e5;
    }
    
    /* Custom Spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .advisory-box {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class AIQuantApp:
    """Main application class"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.featurizer = Featurizer(config.FEATURE_CONFIG)
        self.lstm_model = LSTMModel(config.LSTM_CONFIG)
        self.transformer_model = TransformerModel(config.TRANSFORMER_CONFIG)
        self.predictor = Predictor(config.RISK_CONFIG)
        self.advisory_generator = AdvisoryGenerator(config.ADVISORY_RULES)
        # Initialize LLM advisor only if API key is configured
        try:
            self.llm_advisor = LLMAdvisor()
        except Exception:
            class _LLMStub:
                def is_available(self):
                    return False
                available_models = {}
                max_tokens = 500
                temperature = 0.7
            self.llm_advisor = _LLMStub()
        self.signal_generator = SignalGenerator()
        self.backtester = Backtester(config.BACKTEST_CONFIG)
        self.analyzer = BacktestAnalyzer()
        
    def run(self):
        """Run the main application"""
        # Header
        st.markdown('<div class="main-header">🤖 AI Quant Stock Predictor</div>', unsafe_allow_html=True)
        st.markdown("### Advanced LSTM & Transformer Models for Thai Stock Price Prediction")
        
        # Sidebar inputs
        with st.sidebar:
            st.header("📊 Input Parameters")
            
            # Thai stocks with current prices
            st.markdown("**🇹🇭 Thai Stocks (Current Prices)**")
            thai_stocks = {
                "SET": "SET.BK",  # SET Index
                "SET50": "SET50.BK",  # SET50 Index  
                "PTT": "PTT.BK",  # PTT Public Company Limited
                "SCB": "SCB.BK",  # Siam Commercial Bank
                "KBANK": "KBANK.BK",  # Kasikorn Bank
                "CPALL": "CPALL.BK",  # CP All Public Company Limited
                "ADVANC": "ADVANC.BK",  # Advanced Info Service
                "AOT": "AOT.BK",  # Airports of Thailand
                "BDMS": "BDMS.BK",  # Bangkok Dusit Medical Services
                "CPF": "CPF.BK"  # Charoen Pokphand Foods
            }
            
            # Display current prices for Thai stocks
            for name, symbol_code in thai_stocks.items():
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol_code)
                    info = ticker.info
                    current_price = info.get('regularMarketPrice', 'N/A')
                    change = info.get('regularMarketChange', 0)
                    change_pct = info.get('regularMarketChangePercent', 0)
                    
                    if current_price != 'N/A':
                        color = "🟢" if change >= 0 else "🔴"
                        st.markdown(f"{color} **{name}**: {current_price:.2f} ({change:+.2f}, {change_pct:+.2f}%)")
                except:
                    st.markdown(f"❌ **{name}**: Price unavailable")
            
            st.markdown("---")
            
            # Symbol input with autocomplete suggestions
            symbol = st.text_input(
                "Stock Symbol",
                value="PTT.BK",
                help="Enter stock symbol (e.g., PTT.BK, SCB.BK, AAPL, MSFT)"
            )
            
            # Popular symbols quick select
            st.markdown("**Popular Symbols:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("PTT.BK", use_container_width=True):
                    symbol = "PTT.BK"
                if st.button("SCB.BK", use_container_width=True):
                    symbol = "SCB.BK"
                if st.button("SET.BK", use_container_width=True):
                    symbol = "SET.BK"
            with col2:
                if st.button("KBANK.BK", use_container_width=True):
                    symbol = "KBANK.BK"
                if st.button("CPALL.BK", use_container_width=True):
                    symbol = "CPALL.BK"
                if st.button("AAPL", use_container_width=True):
                    symbol = "AAPL"
                if st.button("TSLA", use_container_width=True):
                    symbol = "TSLA"
            
            st.markdown("---")
            
            # Horizon selection
            horizon_options = config.UI_CONFIG['default_horizons'] + ['Custom']
            horizon_choice = st.selectbox("Prediction Horizon", horizon_options)
            
            if horizon_choice == 'Custom':
                horizon_days = st.number_input("Custom Horizon (days)", min_value=1, max_value=365, value=30)
            else:
                horizon_days = horizon_choice
            
            # Target return
            target_options = config.UI_CONFIG['default_targets'] + ['Custom']
            target_choice = st.selectbox("Target Return (%)", target_options)
            
            if target_choice == 'Custom':
                target_return_pct = st.number_input("Custom Target (%)", min_value=0.1, max_value=100.0, value=5.0)
            else:
                target_return_pct = target_choice
            
            # Backtest scheme
            backtest_scheme = st.selectbox(
                "Backtest Scheme",
                ["A (Long when pred ≥ 0)", "B (Long when pred ≥ target)", "C (Long when prob ≥ 60%)"]
            )
            scheme_letter = backtest_scheme[0]
            
            # LLM Settings
            st.markdown("---")
            st.markdown("**🤖 AI Advisor Settings**")
            use_llm = st.checkbox("Enable LLM Advisor", value=self.llm_advisor.is_available())
            
            if use_llm:
                # Model selection
                selected_model = st.selectbox(
                    "Select AI Model",
                    options=list(self.llm_advisor.available_models.keys()),
                    index=0,  # Default to "Auto (Best)"
                    help="Choose the AI model for generating advice. 'Auto (Best)' uses OpenRouter's automatic model selection."
                )
                
                # Advanced settings
                with st.expander("Advanced Settings"):
                    max_tokens = st.slider("Max Tokens", 100, 1000, 500, help="Maximum response length")
                    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, help="Response creativity (0=deterministic, 1=creative)")
                    
                    # Update advisor settings
                    self.llm_advisor.max_tokens = max_tokens
                    self.llm_advisor.temperature = temperature
            else:
                selected_model = None
            
            if not self.llm_advisor.is_available():
                st.warning("⚠️ LLM not configured. Set OPENROUTER_API_KEY in .env file")
                st.markdown("**Setup:** Copy `env_example.txt` to `.env` and add your API key")
            
            # Action buttons
            st.markdown("---")
            train_button = st.button("🚀 Train & Compare Models", type="primary", use_container_width=True)
            predict_button = st.button("🔮 Quick Predict", type="secondary", use_container_width=True)
        
        # Main content area
        if train_button:
            self.run_training_pipeline(symbol, horizon_days, target_return_pct, scheme_letter, use_llm, selected_model)
        elif predict_button:
            self.run_prediction_only(symbol, horizon_days, target_return_pct, scheme_letter, use_llm, selected_model)
        else:
            self.show_welcome_page()
    
    def run_training_pipeline(self, symbol, horizon_days, target_return_pct, scheme_letter, use_llm=False, selected_model=None):
        """Run complete training and evaluation pipeline"""
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load data
            status_text.text("📥 Loading data...")
            progress_bar.progress(10)
            
            data = self.data_loader.fetch_ohlcv(symbol)
            validation = self.data_loader.validate_data(data)
            
            if not validation['is_valid']:
                for warning in validation['warnings']:
                    st.warning(f"⚠️ {warning}")
            
            # Clean data
            data = self.data_loader.clean_data(data)
            data_summary = self.data_loader.get_data_summary(data)
            
            # Display data summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Period", f"{data_summary['years_covered']:.1f} years")
            with col2:
                st.metric("Trading Days", f"{data_summary['trading_days']:,}")
            with col3:
                st.metric("Current Price", f"${data_summary['price_range']['current']:.2f}")
            with col4:
                st.metric("Price Range", f"${data_summary['price_range']['min']:.2f} - ${data_summary['price_range']['max']:.2f}")
            
            # Step 2: Feature engineering
            status_text.text("🔧 Engineering features...")
            progress_bar.progress(25)
            
            X_train, y_train, X_val, y_val, X_test, y_test, metadata = self.featurizer.make_supervised(
                data, config.DATA_CONFIG['lookback_window'], horizon_days, 'price'
            )
            
            st.success(f"✅ Created {metadata['n_features']} features from {metadata['n_samples']} samples")
            
            # Step 3: Train models
            status_text.text("🤖 Training LSTM model...")
            progress_bar.progress(40)
            
            lstm_model, lstm_history = self.lstm_model.train(X_train, y_train, X_val, y_val)
            
            status_text.text("🤖 Training Transformer model...")
            progress_bar.progress(60)
            
            transformer_model, trans_history = self.transformer_model.train(X_train, y_train, X_val, y_val)
            
            # Step 4: Evaluate models
            status_text.text("📊 Evaluating models...")
            progress_bar.progress(75)
            
            # LSTM evaluation
            lstm_val_pred = self.lstm_model.predict(X_val)
            lstm_test_pred = self.lstm_model.predict(X_test)
            
            lstm_val_metrics = ModelEvaluator.calculate_metrics(y_val, lstm_val_pred)
            lstm_test_metrics = ModelEvaluator.calculate_metrics(y_test, lstm_test_pred)
            
            # Transformer evaluation
            trans_val_pred = self.transformer_model.predict(X_val)
            trans_test_pred = self.transformer_model.predict(X_test)
            
            trans_val_metrics = ModelEvaluator.calculate_metrics(y_val, trans_val_pred)
            trans_test_metrics = ModelEvaluator.calculate_metrics(y_test, trans_test_pred)
            
            # Step 5: Model selection
            status_text.text("🏆 Selecting best model...")
            progress_bar.progress(85)
            
            winner, reason = ModelSelector.select_best_model(lstm_val_metrics, trans_val_metrics)
            
            # Step 6: Generate predictions and advisory
            status_text.text("🔮 Generating predictions...")
            progress_bar.progress(95)
            
            # Use winner model for prediction
            if winner == 'lstm':
                best_model = self.lstm_model
                best_pred = lstm_test_pred
                best_metrics = lstm_test_metrics
            else:
                best_model = self.transformer_model
                best_pred = trans_test_pred
                best_metrics = trans_test_metrics
            
            # Calculate risk metrics
            residuals = y_test - best_pred
            current_price = data['close'].iloc[-1]
            
            # Make forecast
            latest_sequence = X_test[-1:].reshape(1, -1, X_test.shape[-1])
            forecast = self.predictor.forecast(best_model, latest_sequence[0], horizon_days)
            predicted_price = forecast['y_hat']
            predicted_return = self.predictor.calculate_predicted_return(predicted_price, current_price, 'price')
            
            # Risk calculations
            pi_stats = self.predictor.calculate_prediction_interval(residuals)
            hit_probability = self.predictor.calculate_hit_probability(
                predicted_return, target_return_pct, pi_stats['residual_std']
            )
            expected_return = self.predictor.calculate_expected_return(
                predicted_return, hit_probability, target_return_pct
            )
            
            # Generate advisory
            advisory = self.advisory_generator.generate_advisory(
                predicted_return, target_return_pct, hit_probability,
                expected_return['expected_return']
            )
            
            # Step 7: Backtesting
            status_text.text("📈 Running backtest...")
            progress_bar.progress(100)
            
            # Create predictions dataframe for backtesting
            predictions_df = pd.DataFrame({
                'predicted_return': [predicted_return] * len(data),
                'hit_probability': [hit_probability] * len(data)
            }, index=data.index)
            
            signals = self.signal_generator.generate_signals(
                predictions_df, data['close'], scheme_letter, target_return_pct
            )
            
            trades_df, equity_df, backtest_metrics = self.backtester.run_backtest(
                data['close'], signals, horizon_days
            )
            
            # Generate LLM advice if enabled
            llm_advice = None
            if use_llm:
                try:
                    status_text.text("🤖 Generating AI advice...")
                    # Get the actual model name from the selection
                    model_name = self.llm_advisor.available_models.get(selected_model, 'openrouter/auto')
                    llm_advice = self.llm_advisor.generate_human_advice(
                        symbol=symbol,
                        predicted_return=predicted_return,
                        target_return=target_return_pct,
                        hit_probability=hit_probability,
                        expected_return=expected_return['expected_return'],
                        model_performance=best_metrics,
                        backtest_metrics=backtest_metrics,
                        market_context="general",
                        selected_model=model_name
                    )
                except Exception as e:
                    st.warning(f"LLM advice generation failed: {e}")
                    llm_advice = None
            
            # Display results
            self.display_results(
                lstm_val_metrics, lstm_test_metrics, trans_val_metrics, trans_test_metrics,
                winner, reason, predicted_price, predicted_return, hit_probability,
                expected_return, advisory, backtest_metrics, trades_df, equity_df,
                lstm_history, trans_history, y_test, best_pred, llm_advice
            )
            
            status_text.text("✅ Analysis complete!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            status_text.text("❌ Analysis failed")
    
    def run_prediction_only(self, symbol, horizon_days, target_return_pct, scheme_letter, use_llm=False, selected_model=None):
        """Run prediction only (assumes models are already trained)"""
        st.info("🔮 Quick Predict mode - This feature requires pre-trained models. Please use 'Train & Compare Models' first.")
        
        # Show quick prediction interface
        st.markdown("## 🔮 Quick Prediction")
        st.markdown("For a full analysis with model training, please use the 'Train & Compare Models' button.")
        
        # Show example with mock data
        if st.button("Show Example with Mock Data"):
            self.run_mock_example(symbol, horizon_days, target_return_pct, use_llm, selected_model)
    
    def run_mock_example(self, symbol, horizon_days, target_return_pct, use_llm=False):
        """Run example with mock data"""
        st.markdown("## 🎯 Mock Data Example")
        
        # Create mock data
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        np.random.seed(42)
        n_days = 1000
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate mock price data
        initial_price = 100
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create mock OHLCV
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(dates)))
        data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        data['volume'] = np.random.randint(1000000, 10000000, len(dates))
        data = data.dropna()
        
        # Mock prediction results
        current_price = float(data['close'].iloc[-1])
        predicted_price = current_price * (1 + np.random.normal(0.02, 0.05))  # 2% expected return with 5% volatility
        predicted_return = ((predicted_price - current_price) / current_price) * 100
        hit_probability = max(0.1, min(0.9, 0.5 + predicted_return / 20))  # Simple probability calculation
        
        # Display mock results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}")
        with col3:
            st.metric("Predicted Return", f"{predicted_return:.2f}%")
        with col4:
            st.metric("Hit Probability", f"{hit_probability:.1%}")
        
        # Generate mock advisory
        advisory = self.advisory_generator.generate_advisory(
            predicted_return, target_return_pct, hit_probability, predicted_return
        )
        
        # Show advisory
        st.markdown("### 💡 Mock Investment Advisory")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🇹🇭 Thai")
            st.markdown(f'<div class="advisory-box">{advisory["thai"]}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🇺🇸 English")
            st.markdown(f'<div class="advisory-box">{advisory["english"]}</div>', unsafe_allow_html=True)
        
        st.info("ℹ️ This is a mock example. For real analysis, use 'Train & Compare Models' with actual stock data.")
    
    def show_welcome_page(self):
        """Show welcome page with instructions"""
        st.markdown("""
        ## Welcome to AI Quant Stock Predictor! 🚀
        
        This advanced system uses LSTM and Transformer models to predict stock prices and generate investment insights.
        
        ### 🎯 Key Features:
        - **🤖 Dual AI Models**: LSTM vs Transformer comparison with automatic selection
        - **📊 Advanced Analytics**: 50+ technical indicators and pattern recognition
        - **⚖️ Risk Assessment**: Prediction intervals, hit probabilities, and expected returns
        - **📈 Backtesting Engine**: Multiple trading schemes with realistic transaction costs
        - **🌐 Bilingual Advisory**: Thai and English investment guidance
        - **🧠 LLM Integration**: AI-powered human-readable advice (OpenRouter API)
        
        ### 🚀 How to Use:
        1. **📝 Enter Stock Symbol**: Use any valid symbol (AAPL, GOOGL, MSFT, TSLA)
        2. **⏰ Set Prediction Horizon**: Choose 1, 5, 7, 14, 30 days or custom
        3. **🎯 Define Target Return**: Set your desired return percentage
        4. **📊 Select Backtest Scheme**: Choose your trading strategy
        5. **🤖 Enable LLM Advisor**: Toggle AI-powered advice (requires API key)
        6. **▶️ Click "Train & Compare Models"** to start analysis
        
        ### 🏆 Model Selection Criteria:
        - **Primary**: Lowest RMSE on validation set
        - **Secondary**: If RMSE difference < 2%, use MAE
        - **Tertiary**: If still tied, use MAPE
        
        ### 📊 Risk Metrics:
        - **Expected Return (ER)**: Risk-adjusted expected return
        - **Prediction Interval (PI)**: 95% confidence range
        - **Hit Probability**: Likelihood of achieving target return
        - **Sharpe Ratio**: Risk-adjusted performance measure
        - **Maximum Drawdown**: Worst peak-to-trough decline
        
        ### 🤖 LLM Advisor Setup:
        To enable AI-powered human-readable advice:
        1. Copy `env_example.txt` to `.env`
        2. Add your OpenRouter API key
        3. Enable "LLM Advisor" in the sidebar
        
        ⚠️ **Disclaimer**: This tool is for educational purposes only and does not constitute investment advice.
        """)
        
        # Sample analysis preview
        st.markdown("### Sample Analysis Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Performance Comparison**")
            sample_data = {
                'Model': ['LSTM', 'Transformer'],
                'RMSE': [2.45, 2.38],
                'MAE': [1.89, 1.92],
                'MAPE': [3.2, 3.1],
                'R²': [0.85, 0.87]
            }
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        
        with col2:
            st.markdown("**Prediction Results**")
            st.metric("Predicted Price", "$156.78")
            st.metric("Predicted Return", "3.2%")
            st.metric("Hit Probability", "68%")
            st.metric("Expected Return", "2.8%")
    
    def display_results(self, lstm_val_metrics, lstm_test_metrics, trans_val_metrics, 
                       trans_test_metrics, winner, reason, predicted_price, predicted_return,
                       hit_probability, expected_return, advisory, backtest_metrics,
                       trades_df, equity_df, lstm_history, trans_history, y_test, best_pred, llm_advice=None):
        """Display comprehensive results"""
        
        # Model comparison
        st.markdown("## 📊 Model Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### LSTM Model")
            lstm_data = {
                'Metric': ['RMSE', 'MAE', 'MAPE', 'R²'],
                'Validation': [lstm_val_metrics['RMSE'], lstm_val_metrics['MAE'], 
                              lstm_val_metrics['MAPE'], lstm_val_metrics['R2']],
                'Test': [lstm_test_metrics['RMSE'], lstm_test_metrics['MAE'], 
                        lstm_test_metrics['MAPE'], lstm_test_metrics['R2']]
            }
            st.dataframe(pd.DataFrame(lstm_data), use_container_width=True)
        
        with col2:
            st.markdown("### Transformer Model")
            trans_data = {
                'Metric': ['RMSE', 'MAE', 'MAPE', 'R²'],
                'Validation': [trans_val_metrics['RMSE'], trans_val_metrics['MAE'], 
                              trans_val_metrics['MAPE'], trans_val_metrics['R2']],
                'Test': [trans_test_metrics['RMSE'], trans_test_metrics['MAE'], 
                        trans_test_metrics['MAPE'], trans_test_metrics['R2']]
            }
            st.dataframe(pd.DataFrame(trans_data), use_container_width=True)
        
        # Winner announcement
        st.markdown(f"## 🏆 Selected Model: {winner.upper()}")
        st.info(f"**Reason**: {reason}")
        
        # Prediction results
        st.markdown("## 🔮 Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predicted Price", f"${predicted_price:.2f}")
        with col2:
            st.metric("Predicted Return", f"{predicted_return:.2f}%")
        with col3:
            st.metric("Hit Probability", f"{hit_probability:.1%}")
        with col4:
            st.metric("Expected Return", f"{expected_return['expected_return']:.2f}%")
        
        # Advisory
        st.markdown("## 💡 Investment Advisory")
        
        # Show LLM advice if available
        if llm_advice and llm_advice.get('llm_enhanced', False):
            st.markdown("### 🤖 AI-Powered Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🇹🇭 Thai")
                st.markdown(f'<div class="advisory-box">{llm_advice["thai"]}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🇺🇸 English")
                st.markdown(f'<div class="advisory-box">{llm_advice["english"]}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📊 System Analysis (Fallback)")
        else:
            st.markdown("### 📊 System Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🇹🇭 Thai")
            st.markdown(f'<div class="advisory-box">{advisory["thai"]}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🇺🇸 English")
            st.markdown(f'<div class="advisory-box">{advisory["english"]}</div>', unsafe_allow_html=True)
        
        # Backtest results
        st.markdown("## 📈 Backtest Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CAGR", f"{backtest_metrics['CAGR']:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{backtest_metrics['Sharpe']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{backtest_metrics['Max_Drawdown']:.2f}%")
        with col4:
            st.metric("Win Rate", f"{backtest_metrics['Win_Rate']:.1f}%")
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volatility", f"{backtest_metrics['Volatility']:.2f}%")
        with col2:
            st.metric("Profit Factor", f"{backtest_metrics['Profit_Factor']:.2f}")
        with col3:
            st.metric("Total Trades", f"{backtest_metrics['N_Trades']}")
        with col4:
            st.metric("Turnover", f"{backtest_metrics['Turnover']:.1f}")
        
        # Charts
        self.create_charts(lstm_history, trans_history, y_test, best_pred, equity_df, trades_df)
    
    def create_charts(self, lstm_history, trans_history, y_test, best_pred, equity_df, trades_df):
        """Create visualization charts"""
        
        # Learning curves
        st.markdown("## 📈 Learning Curves")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(y=lstm_history.history['loss'], name='Training Loss', line=dict(color='blue')))
            fig_lstm.add_trace(go.Scatter(y=lstm_history.history['val_loss'], name='Validation Loss', line=dict(color='red')))
            fig_lstm.update_layout(title='LSTM Learning Curve', xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig_lstm, use_container_width=True)
        
        with col2:
            fig_trans = go.Figure()
            fig_trans.add_trace(go.Scatter(y=trans_history.history['loss'], name='Training Loss', line=dict(color='blue')))
            fig_trans.add_trace(go.Scatter(y=trans_history.history['val_loss'], name='Validation Loss', line=dict(color='red')))
            fig_trans.update_layout(title='Transformer Learning Curve', xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig_trans, use_container_width=True)
        
        # Prediction vs Actual
        st.markdown("## 🎯 Prediction vs Actual")
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=y_test, name='Actual', mode='lines', line=dict(color='blue')))
        fig_pred.add_trace(go.Scatter(y=best_pred, name='Predicted', mode='lines', line=dict(color='red')))
        fig_pred.update_layout(title='Test Set: Actual vs Predicted', xaxis_title='Time', yaxis_title='Price')
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Equity curve
        if not equity_df.empty:
            st.markdown("## 💰 Equity Curve")
            
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=equity_df.index, 
                y=equity_df['portfolio_value'], 
                name='Portfolio Value',
                line=dict(color='green')
            ))
            fig_equity.update_layout(title='Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Value ($)')
            st.plotly_chart(fig_equity, use_container_width=True)
        
        # Trade log
        if not trades_df.empty:
            st.markdown("## 📋 Recent Trades")
            recent_trades = trades_df.tail(10)
            st.dataframe(recent_trades, use_container_width=True)
        
        # Legal Disclaimer
        st.markdown("---")
        st.markdown("### ⚠️ Important Legal Disclaimer")
        st.markdown("""
        <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
        <h4 style="color: #c62828; margin-top: 0;">🚨 Investment Risk Warning</h4>
        <p style="color: #424242; margin-bottom: 10px;">
        <strong>This application is for educational and research purposes only.</strong> 
        It is NOT intended for actual trading or investment decisions.
        </p>
        <ul style="color: #424242; margin-bottom: 10px;">
        <li>All predictions and analysis are for informational purposes only</li>
        <li>Past performance does not guarantee future results</li>
        <li>Stock prices are highly volatile and unpredictable</li>
        <li>AI models can make errors and should not be relied upon</li>
        <li>Always consult with qualified financial professionals</li>
        </ul>
        <p style="color: #424242; margin-bottom: 0;">
        <strong>By using this application, you acknowledge and accept all risks.</strong>
        The authors and contributors are not responsible for any financial losses.
        </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function"""
    app = AIQuantApp()
    app.run()

if __name__ == "__main__":
    main()
