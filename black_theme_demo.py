import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# Page config
st.set_page_config(
    page_title="AI Quant - Black Theme Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Professional Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        font-family: 'Inter', sans-serif;
        color: #ffffff;
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
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🤖 AI Quant Stock Predictor</div>', unsafe_allow_html=True)
    st.markdown("### Dark Professional Theme with Thai Stocks")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Thai Stock Prices")
        
        # Thai stocks with current prices
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
                ticker = yf.Ticker(symbol_code)
                info = ticker.info
                current_price = info.get('regularMarketPrice', 'N/A')
                change = info.get('regularMarketChange', 0)
                change_pct = info.get('regularMarketChangePercent', 0)
                
                if current_price != 'N/A':
                    color = "🟢" if change >= 0 else "🔴"
                    st.markdown(f"{color} **{name}**: {current_price:.2f} ({change:+.2f}, {change_pct:+.2f}%)")
                else:
                    st.markdown(f"❌ **{name}**: Price unavailable")
            except Exception as e:
                st.markdown(f"❌ **{name}**: Error loading price")
        
        st.markdown("---")
        
        # Input controls
        symbol = st.text_input("Stock Symbol", value="PTT.BK")
        horizon_days = st.slider("Prediction Horizon (days)", 1, 30, 7)
        target_return = st.number_input("Target Return (%)", -10.0, 20.0, 5.0, 0.1)
        
        # Quick select buttons
        st.markdown("**Quick Select:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("PTT.BK", use_container_width=True):
                symbol = "PTT.BK"
            if st.button("SCB.BK", use_container_width=True):
                symbol = "SCB.BK"
        with col2:
            if st.button("SET.BK", use_container_width=True):
                symbol = "SET.BK"
            if st.button("KBANK.BK", use_container_width=True):
                symbol = "KBANK.BK"
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Price", "45.50", "2.30", "5.33%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Price", "47.20", "1.70", "3.74%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Hit Probability", "78.5%", "12.3%", "18.6%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Expected Return", "3.2%", "0.8%", "33.3%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Advisory section
    st.markdown('<div class="advisory-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Investment Advisory")
    st.markdown("""
    **System Analysis:**
    - Predicted return: 3.2% (Target: 5.0%)
    - Hit probability: 78.5% (Above threshold)
    - Risk level: Medium
    - Recommendation: **HOLD** - Wait for better entry point
    
    **AI Advisor:**
    Based on technical analysis and market conditions, PTT.BK shows moderate bullish signals. 
    The model suggests waiting for a pullback to 44.50-45.00 range for better risk-reward ratio.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts section
    st.markdown("### 📈 Performance Charts")
    
    # Sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    prices = 45 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=prices, 
        mode='lines',
        name='Price',
        line=dict(color='#00d4ff', width=3)
    ))
    
    fig.update_layout(
        title="Stock Price Trend",
        xaxis_title="Date",
        yaxis_title="Price (THB)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='#333333')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.markdown("### 📊 Model Performance")
    
    performance_data = {
        'Model': ['LSTM', 'Transformer', 'Ensemble'],
        'MAE': [1.23, 1.45, 1.12],
        'RMSE': [1.67, 1.89, 1.54],
        'MAPE': [2.71, 3.19, 2.46],
        'R²': [0.78, 0.72, 0.81]
    }
    
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True)
    
    # Backtest results
    st.markdown("### 🎯 Backtest Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Performance Metrics:**")
        st.markdown("- CAGR: 12.5%")
        st.markdown("- Sharpe Ratio: 1.34")
        st.markdown("- Max Drawdown: -8.2%")
        st.markdown("- Win Rate: 68.5%")
    
    with col2:
        st.markdown("**Risk Metrics:**")
        st.markdown("- Volatility: 15.2%")
        st.markdown("- Profit Factor: 1.89")
        st.markdown("- Hit Ratio: 72.3%")
        st.markdown("- Total Trades: 156")

if __name__ == "__main__":
    main()
