"""
UI Demo for AI Quant Stock Predictor - Minimal White Theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="AI Quant UI Demo",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Minimal White Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #fafafa;
        border-right: 1px solid #e5e5e5;
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
    }
    
    .css-1d391kg .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
    }
    
    .css-1d391kg .stButton > button {
        background-color: #ffffff;
        color: #1a1a1a;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .css-1d391kg .stButton > button:hover {
        background-color: #f5f5f5;
        border-color: #d4d4d4;
        transform: translateY(-1px);
    }
    
    /* Primary Button */
    .css-1d391kg .stButton > button[kind="primary"] {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #1a1a1a;
    }
    
    .css-1d391kg .stButton > button[kind="primary"]:hover {
        background-color: #333333;
        border-color: #333333;
    }
    
    /* Secondary Button */
    .css-1d391kg .stButton > button[kind="secondary"] {
        background-color: #f5f5f5;
        color: #1a1a1a;
        border: 1px solid #e5e5e5;
    }
    
    .css-1d391kg .stButton > button[kind="secondary"]:hover {
        background-color: #e5e5e5;
        border-color: #d4d4d4;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Advisory Boxes */
    .advisory-box {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .advisory-box h3 {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .advisory-box p {
        color: #4a4a4a;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    /* Warning Box */
    .warning-box {
        background-color: #fff8e1;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.1);
    }
    
    /* Success Box */
    .success-box {
        background-color: #f1f8e9;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);
    }
    
    /* Info Box */
    .info-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
    }
    
    /* Tables */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Charts */
    .js-plotly-plot {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #1a1a1a;
        border-radius: 4px;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
    }
    
    /* Checkbox */
    .stCheckbox > div > div > div {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 4px;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #1a1a1a;
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

def main():
    """Main demo function"""
    # Header
    st.markdown('<div class="main-header">🎨 AI Quant UI Demo</div>', unsafe_allow_html=True)
    st.markdown("### Minimal White Theme Preview")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Demo Controls")
        
        # Input controls
        symbol = st.text_input("Stock Symbol", value="AAPL")
        horizon = st.selectbox("Prediction Horizon", [1, 5, 7, 14, 30])
        target_return = st.slider("Target Return (%)", 1.0, 20.0, 5.0)
        
        # Model selection
        st.markdown("**🤖 AI Model**")
        model = st.selectbox(
            "Select Model",
            ["Auto (Best)", "GPT-4o Mini", "GPT-3.5 Turbo", "Claude 3 Haiku", "Claude 3 Sonnet"]
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_tokens = st.slider("Max Tokens", 100, 1000, 500)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        # Buttons
        st.markdown("---")
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.success("Analysis started!")
        
        if st.button("🔮 Quick Predict", type="secondary", use_container_width=True):
            st.info("Quick prediction mode")
    
    # Main content
    st.markdown("## 📈 Sample Analysis Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", "$156.78", "2.34%")
    with col2:
        st.metric("Predicted Price", "$162.45", "3.62%")
    with col3:
        st.metric("Hit Probability", "68%", "12%")
    with col4:
        st.metric("Expected Return", "4.2%", "0.8%")
    
    # Model comparison
    st.markdown("## 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### LSTM Model")
        lstm_data = {
            'Metric': ['RMSE', 'MAE', 'MAPE', 'R²'],
            'Validation': [2.45, 1.89, 3.2, 0.85],
            'Test': [2.38, 1.92, 3.1, 0.87]
        }
        st.dataframe(pd.DataFrame(lstm_data), use_container_width=True)
    
    with col2:
        st.markdown("### Transformer Model")
        trans_data = {
            'Metric': ['RMSE', 'MAE', 'MAPE', 'R²'],
            'Validation': [2.38, 1.92, 3.1, 0.87],
            'Test': [2.45, 1.89, 3.2, 0.85]
        }
        st.dataframe(pd.DataFrame(trans_data), use_container_width=True)
    
    # Winner
    st.markdown("## 🏆 Selected Model: Transformer")
    st.info("**Reason**: Transformer has lower RMSE (2.38 vs 2.45)")
    
    # Advisory
    st.markdown("## 💡 Investment Advisory")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🇹🇭 Thai")
        st.markdown(f'''
        <div class="advisory-box">
        <h3>📈 การวิเคราะห์เชิงบวก</h3>
        <p><strong>ผลตอบแทนที่คาดการณ์:</strong> 3.62%</p>
        <p><strong>เป้าหมาย:</strong> 5.00%</p>
        <p><strong>ความน่าจะเป็นถึงเป้า:</strong> 68%</p>
        <p><strong>ผลตอบแทนที่คาดหวัง:</strong> 4.2%</p>
        <p><strong>คำแนะนำ:</strong> ตามการวิเคราะห์ มีโอกาสสูงที่จะบรรลุเป้าหมายผลตอบแทน อย่างไรก็ตาม การลงทุนมีความเสี่ยง โปรดพิจารณาให้รอบคอบ</p>
        <p><strong>หมายเหตุ:</strong> ข้อมูลนี้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🇺🇸 English")
        st.markdown(f'''
        <div class="advisory-box">
        <h3>📈 Positive Analysis</h3>
        <p><strong>Predicted Return:</strong> 3.62%</p>
        <p><strong>Target Return:</strong> 5.00%</p>
        <p><strong>Hit Probability:</strong> 68%</p>
        <p><strong>Expected Return:</strong> 4.2%</p>
        <p><strong>Recommendation:</strong> Based on analysis, there's a high probability of achieving the target return. However, investing involves risks, please consider carefully.</p>
        <p><strong>Disclaimer:</strong> This information is for educational purposes only, not investment advice.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Backtest results
    st.markdown("## 📈 Backtest Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CAGR", "12.5%", "2.1%")
    with col2:
        st.metric("Sharpe Ratio", "1.85", "0.23")
    with col3:
        st.metric("Max Drawdown", "-8.2%", "1.5%")
    with col4:
        st.metric("Win Rate", "68%", "5%")
    
    # Charts
    st.markdown("## 📊 Performance Charts")
    
    # Sample data for charts
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    # Price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=dates, y=prices, name='Price', line=dict(color='#1a1a1a')))
    fig_price.update_layout(
        title='Stock Price Over Time',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white'
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Performance metrics chart
    metrics_data = {
        'Model': ['LSTM', 'Transformer', 'LSTM', 'Transformer'],
        'Metric': ['RMSE', 'RMSE', 'MAE', 'MAE'],
        'Value': [2.45, 2.38, 1.89, 1.92]
    }
    
    fig_metrics = px.bar(
        pd.DataFrame(metrics_data), 
        x='Model', 
        y='Value', 
        color='Metric',
        title='Model Performance Comparison',
        template='plotly_white'
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Sample table
    st.markdown("## 📋 Recent Trades")
    
    trades_data = {
        'Date': ['2024-01-15', '2024-01-14', '2024-01-13', '2024-01-12', '2024-01-11'],
        'Action': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
        'Price': [156.78, 154.32, 152.45, 151.23, 149.87],
        'Return': ['+2.34%', '+1.89%', '+1.23%', '+0.87%', '+0.45%'],
        'P&L': ['+$234', '+$189', '+$123', '+$87', '+$45']
    }
    
    st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
    
    # Status messages
    st.markdown("## 📝 System Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Data loaded successfully")
    with col2:
        st.info("ℹ️ Model training completed")
    with col3:
        st.warning("⚠️ LLM not configured")

if __name__ == "__main__":
    main()
