"""
Localization and Thai/English Support

This module provides bilingual support for the AI Quant application,
including Thai/English summaries, disclaimers, and UI text.
"""

import streamlit as st
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional

class LocalizationManager:
    """Manage Thai/English localization"""
    
    def __init__(self):
        self.language = st.session_state.get('language', 'th')
        self.texts = self._load_texts()
    
    def _load_texts(self) -> Dict[str, Dict[str, str]]:
        """Load localized text strings"""
        return {
            'th': {
                # Main UI
                'app_title': 'AI Quant Stock Predictor',
                'app_subtitle': 'ระบบทำนายราคาหุ้นด้วย AI',
                'sidebar_title': 'การตั้งค่า',
                'symbol_input': 'สัญลักษณ์หุ้น',
                'symbol_help': 'กรอกสัญลักษณ์หุ้น เช่น PTT, SCB, KBANK',
                'horizon_input': 'ระยะเวลาทำนาย (วัน)',
                'target_input': 'เป้าหมายผลตอบแทน (%)',
                'run_prediction': 'ทำนายราคา',
                'run_backtest': 'ทดสอบย้อนหลัง',
                
                # Results
                'prediction_results': 'ผลการทำนาย',
                'backtest_results': 'ผลการทดสอบย้อนหลัง',
                'model_performance': 'ประสิทธิภาพโมเดล',
                'trading_signals': 'สัญญาณการซื้อขาย',
                'risk_metrics': 'ตัวชี้วัดความเสี่ยง',
                
                # Metrics
                'rmse': 'RMSE',
                'mae': 'MAE',
                'mape': 'MAPE (%)',
                'r2': 'R²',
                'hit_probability': 'ความน่าจะเป็นในการชนะ',
                'sharpe_ratio': 'อัตราส่วนชาร์ป',
                'max_drawdown': 'การลดลงสูงสุด',
                'win_rate': 'อัตราชนะ',
                
                # Status messages
                'loading_data': 'กำลังโหลดข้อมูล...',
                'training_model': 'กำลังฝึกโมเดล...',
                'running_backtest': 'กำลังทดสอบย้อนหลัง...',
                'data_loaded': 'โหลดข้อมูลสำเร็จ',
                'model_trained': 'ฝึกโมเดลสำเร็จ',
                'backtest_complete': 'ทดสอบย้อนหลังเสร็จสิ้น',
                
                # Errors
                'error_loading_data': 'เกิดข้อผิดพลาดในการโหลดข้อมูล',
                'error_training': 'เกิดข้อผิดพลาดในการฝึกโมเดล',
                'error_prediction': 'เกิดข้อผิดพลาดในการทำนาย',
                'error_backtest': 'เกิดข้อผิดพลาดในการทดสอบย้อนหลัง',
                'invalid_symbol': 'สัญลักษณ์หุ้นไม่ถูกต้อง',
                'insufficient_data': 'ข้อมูลไม่เพียงพอ',
                
                # Disclaimers
                'disclaimer_title': 'ข้อจำกัดและข้อควรระวัง',
                'disclaimer_text': '''
                ⚠️ **ข้อจำกัดและข้อควรระวัง**
                
                • **ไม่ใช่คำแนะนำการลงทุน**: ระบบนี้เป็นเครื่องมือการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน
                • **ความเสี่ยงสูง**: การลงทุนในหุ้นมีความเสี่ยงสูง อาจสูญเสียเงินลงทุนได้
                • **ผลลัพธ์ในอดีต**: ผลการทดสอบย้อนหลังไม่รับประกันผลการดำเนินงานในอนาคต
                • **ความไม่แน่นอน**: การทำนายราคาหุ้นมีความไม่แน่นอนสูง
                • **ปรึกษาผู้เชี่ยวชาญ**: ควรปรึกษาผู้เชี่ยวชาญด้านการลงทุนก่อนตัดสินใจ
                • **ใช้วิจารณญาณ**: ใช้ข้อมูลนี้เป็นเพียงข้อมูลประกอบการตัดสินใจ
                ''',
                
                # Download buttons
                'download_forecast': 'ดาวน์โหลดการทำนาย',
                'download_trades': 'ดาวน์โหลดรายการซื้อขาย',
                'download_report': 'ดาวน์โหลดรายงาน',
                'download_data': 'ดาวน์โหลดข้อมูล',
                
                # Model selection
                'model_selection': 'การเลือกโมเดล',
                'best_model': 'โมเดลที่ดีที่สุด',
                'selection_reason': 'เหตุผลในการเลือก',
                'baseline_comparison': 'เปรียบเทียบกับโมเดลพื้นฐาน',
                
                # Time periods
                'today': 'วันนี้',
                'yesterday': 'เมื่อวาน',
                'this_week': 'สัปดาห์นี้',
                'this_month': 'เดือนนี้',
                'this_year': 'ปีนี้',
                'last_30_days': '30 วันที่แล้ว',
                'last_90_days': '90 วันที่แล้ว',
                'last_year': 'ปีที่แล้ว',
                
                # Thai stock symbols
                'thai_stocks': {
                    'PTT': 'ปตท.',
                    'SCB': 'ธนาคารไทยพาณิชย์',
                    'KBANK': 'ธนาคารกสิกรไทย',
                    'CPALL': 'ซีพีออลล์',
                    'ADVANC': 'แอดวานซ์ อินโฟร์ เซอร์วิส',
                    'AOT': 'ท่าอากาศยานไทย',
                    'BDMS': 'บำรุงราษฎร์',
                    'CPF': 'เจริญโภคภัณฑ์อาหาร',
                    'SET': 'ดัชนี SET',
                    'SET50': 'ดัชนี SET50'
                }
            },
            
            'en': {
                # Main UI
                'app_title': 'AI Quant Stock Predictor',
                'app_subtitle': 'AI-Powered Stock Price Prediction System',
                'sidebar_title': 'Settings',
                'symbol_input': 'Stock Symbol',
                'symbol_help': 'Enter stock symbol e.g., PTT, SCB, KBANK',
                'horizon_input': 'Prediction Horizon (days)',
                'target_input': 'Target Return (%)',
                'run_prediction': 'Run Prediction',
                'run_backtest': 'Run Backtest',
                
                # Results
                'prediction_results': 'Prediction Results',
                'backtest_results': 'Backtest Results',
                'model_performance': 'Model Performance',
                'trading_signals': 'Trading Signals',
                'risk_metrics': 'Risk Metrics',
                
                # Metrics
                'rmse': 'RMSE',
                'mae': 'MAE',
                'mape': 'MAPE (%)',
                'r2': 'R²',
                'hit_probability': 'Hit Probability',
                'sharpe_ratio': 'Sharpe Ratio',
                'max_drawdown': 'Max Drawdown',
                'win_rate': 'Win Rate',
                
                # Status messages
                'loading_data': 'Loading data...',
                'training_model': 'Training model...',
                'running_backtest': 'Running backtest...',
                'data_loaded': 'Data loaded successfully',
                'model_trained': 'Model trained successfully',
                'backtest_complete': 'Backtest completed',
                
                # Errors
                'error_loading_data': 'Error loading data',
                'error_training': 'Error training model',
                'error_prediction': 'Error making prediction',
                'error_backtest': 'Error running backtest',
                'invalid_symbol': 'Invalid stock symbol',
                'insufficient_data': 'Insufficient data',
                
                # Disclaimers
                'disclaimer_title': 'Limitations and Disclaimers',
                'disclaimer_text': '''
                ⚠️ **Limitations and Disclaimers**
                
                • **Not Investment Advice**: This system is for educational purposes only, not investment advice
                • **High Risk**: Stock investing involves high risk and may result in loss of capital
                • **Past Performance**: Backtest results do not guarantee future performance
                • **Uncertainty**: Stock price predictions are highly uncertain
                • **Consult Experts**: Consult investment professionals before making decisions
                • **Use Judgment**: Use this information as supplementary data only
                ''',
                
                # Download buttons
                'download_forecast': 'Download Forecast',
                'download_trades': 'Download Trades',
                'download_report': 'Download Report',
                'download_data': 'Download Data',
                
                # Model selection
                'model_selection': 'Model Selection',
                'best_model': 'Best Model',
                'selection_reason': 'Selection Reason',
                'baseline_comparison': 'Baseline Comparison',
                
                # Time periods
                'today': 'Today',
                'yesterday': 'Yesterday',
                'this_week': 'This Week',
                'this_month': 'This Month',
                'this_year': 'This Year',
                'last_30_days': 'Last 30 Days',
                'last_90_days': 'Last 90 Days',
                'last_year': 'Last Year',
                
                # Thai stock symbols
                'thai_stocks': {
                    'PTT': 'PTT Public Company Limited',
                    'SCB': 'Siam Commercial Bank',
                    'KBANK': 'Kasikorn Bank',
                    'CPALL': 'CP All Public Company Limited',
                    'ADVANC': 'Advanced Info Service',
                    'AOT': 'Airports of Thailand',
                    'BDMS': 'Bangkok Dusit Medical Services',
                    'CPF': 'Charoen Pokphand Foods',
                    'SET': 'SET Index',
                    'SET50': 'SET50 Index'
                }
            }
        }
    
    def get_text(self, key: str, **kwargs) -> str:
        """Get localized text"""
        text = self.texts[self.language].get(key, key)
        
        # Format with kwargs if provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass
        
        return text
    
    def get_thai_stock_name(self, symbol: str) -> str:
        """Get Thai stock name for symbol"""
        return self.texts[self.language]['thai_stocks'].get(symbol, symbol)
    
    def set_language(self, language: str):
        """Set current language"""
        self.language = language
        st.session_state['language'] = language
    
    def format_number(self, value: float, decimals: int = 2) -> str:
        """Format number according to locale"""
        if self.language == 'th':
            # Thai number formatting
            return f"{value:,.{decimals}f}"
        else:
            # English number formatting
            return f"{value:,.{decimals}f}"
    
    def format_percentage(self, value: float, decimals: int = 2) -> str:
        """Format percentage according to locale"""
        if self.language == 'th':
            return f"{value:.{decimals}f}%"
        else:
            return f"{value:.{decimals}f}%"
    
    def format_currency(self, value: float, currency: str = 'THB') -> str:
        """Format currency according to locale"""
        if self.language == 'th':
            return f"฿{value:,.2f}"
        else:
            return f"{currency} {value:,.2f}"


class SummaryGenerator:
    """Generate bilingual summaries"""
    
    def __init__(self, localization_manager: LocalizationManager):
        self.loc = localization_manager
    
    def generate_prediction_summary(self, symbol: str, predictions: Dict, 
                                  model_name: str, horizon_days: int) -> Dict[str, str]:
        """Generate prediction summary in both languages"""
        
        # Get current price and prediction
        current_price = predictions.get('current_price', 0)
        predicted_price = predictions.get('predicted_price', 0)
        predicted_return = predictions.get('predicted_return', 0)
        confidence = predictions.get('confidence', 0)
        
        # Thai summary
        thai_summary = f"""
        📊 **สรุปการทำนายราคาหุ้น {self.loc.get_thai_stock_name(symbol)}**
        
        • **ราคาปัจจุบัน**: {self.loc.format_currency(current_price)}
        • **ราคาทำนาย ({horizon_days} วัน)**: {self.loc.format_currency(predicted_price)}
        • **ผลตอบแทนที่คาดการณ์**: {self.loc.format_percentage(predicted_return)}
        • **ความเชื่อมั่น**: {self.loc.format_percentage(confidence)}
        • **โมเดลที่ใช้**: {model_name}
        • **วันที่ทำนาย**: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        """
        
        # English summary
        english_summary = f"""
        📊 **{symbol} Stock Price Prediction Summary**
        
        • **Current Price**: {self.loc.format_currency(current_price)}
        • **Predicted Price ({horizon_days} days)**: {self.loc.format_currency(predicted_price)}
        • **Expected Return**: {self.loc.format_percentage(predicted_return)}
        • **Confidence Level**: {self.loc.format_percentage(confidence)}
        • **Model Used**: {model_name}
        • **Prediction Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        
        return {
            'thai': thai_summary,
            'english': english_summary
        }
    
    def generate_backtest_summary(self, symbol: str, backtest_results: Dict) -> Dict[str, str]:
        """Generate backtest summary in both languages"""
        
        # Extract key metrics
        total_return = backtest_results.get('total_return', 0)
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
        max_drawdown = backtest_results.get('max_drawdown', 0)
        win_rate = backtest_results.get('win_rate', 0)
        total_trades = backtest_results.get('total_trades', 0)
        
        # Thai summary
        thai_summary = f"""
        📈 **สรุปผลการทดสอบย้อนหลัง {self.loc.get_thai_stock_name(symbol)}**
        
        • **ผลตอบแทนรวม**: {self.loc.format_percentage(total_return)}
        • **อัตราส่วนชาร์ป**: {self.loc.format_number(sharpe_ratio, 3)}
        • **การลดลงสูงสุด**: {self.loc.format_percentage(max_drawdown)}
        • **อัตราชนะ**: {self.loc.format_percentage(win_rate)}
        • **จำนวนการซื้อขาย**: {total_trades:,} ครั้ง
        • **ระยะเวลาทดสอบ**: {backtest_results.get('period', 'N/A')}
        """
        
        # English summary
        english_summary = f"""
        📈 **{symbol} Backtest Results Summary**
        
        • **Total Return**: {self.loc.format_percentage(total_return)}
        • **Sharpe Ratio**: {self.loc.format_number(sharpe_ratio, 3)}
        • **Max Drawdown**: {self.loc.format_percentage(max_drawdown)}
        • **Win Rate**: {self.loc.format_percentage(win_rate)}
        • **Total Trades**: {total_trades:,} trades
        • **Test Period**: {backtest_results.get('period', 'N/A')}
        """
        
        return {
            'thai': thai_summary,
            'english': english_summary
        }
    
    def generate_model_performance_summary(self, model_metrics: Dict, 
                                         baseline_comparison: Optional[Dict] = None) -> Dict[str, str]:
        """Generate model performance summary in both languages"""
        
        rmse = model_metrics.get('RMSE', 0)
        mae = model_metrics.get('MAE', 0)
        mape = model_metrics.get('MAPE', 0)
        r2 = model_metrics.get('R2', 0)
        
        # Thai summary
        thai_summary = f"""
        🤖 **สรุปประสิทธิภาพโมเดล**
        
        • **RMSE**: {self.loc.format_number(rmse, 4)}
        • **MAE**: {self.loc.format_number(mae, 4)}
        • **MAPE**: {self.loc.format_percentage(mape, 2)}
        • **R²**: {self.loc.format_number(r2, 4)}
        """
        
        # Add baseline comparison if available
        if baseline_comparison:
            improvement = baseline_comparison.get('improvement_pct', 0)
            if improvement > 0:
                thai_summary += f"\n• **ดีกว่าโมเดลพื้นฐาน**: {self.loc.format_percentage(improvement, 2)}"
            else:
                thai_summary += f"\n• **ด้อยกว่าโมเดลพื้นฐาน**: {self.loc.format_percentage(abs(improvement), 2)}"
        
        # English summary
        english_summary = f"""
        🤖 **Model Performance Summary**
        
        • **RMSE**: {self.loc.format_number(rmse, 4)}
        • **MAE**: {self.loc.format_number(mae, 4)}
        • **MAPE**: {self.loc.format_percentage(mape, 2)}
        • **R²**: {self.loc.format_number(r2, 4)}
        """
        
        # Add baseline comparison if available
        if baseline_comparison:
            improvement = baseline_comparison.get('improvement_pct', 0)
            if improvement > 0:
                english_summary += f"\n• **Better than Baseline**: {self.loc.format_percentage(improvement, 2)}"
            else:
                english_summary += f"\n• **Worse than Baseline**: {self.loc.format_percentage(abs(improvement), 2)}"
        
        return {
            'thai': thai_summary,
            'english': english_summary
        }


def create_download_buttons(data_dict: Dict[str, Any], file_prefix: str = "ai_quant") -> Dict[str, str]:
    """Create download buttons for various data formats"""
    
    downloads = {}
    
    for name, data in data_dict.items():
        if isinstance(data, pd.DataFrame):
            # CSV download
            csv = data.to_csv(index=False)
            downloads[f"{name}_csv"] = csv
        elif isinstance(data, dict):
            # JSON download
            import json
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            downloads[f"{name}_json"] = json_str
        elif isinstance(data, str):
            # Text download
            downloads[f"{name}_txt"] = data
    
    return downloads


def display_bilingual_summary(summary_dict: Dict[str, str], title: str = "Summary"):
    """Display bilingual summary with tabs"""
    
    tab1, tab2 = st.tabs(["🇹🇭 ไทย", "🇺🇸 English"])
    
    with tab1:
        st.markdown(f"### {title} (ไทย)")
        st.markdown(summary_dict['thai'])
    
    with tab2:
        st.markdown(f"### {title} (English)")
        st.markdown(summary_dict['english'])


def display_disclaimer(localization_manager: LocalizationManager):
    """Display disclaimer in current language"""
    
    st.markdown("---")
    st.markdown(f"### {localization_manager.get_text('disclaimer_title')}")
    st.markdown(localization_manager.get_text('disclaimer_text'))
    
    # Add language toggle
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🇹🇭 ไทย", key="lang_th"):
            localization_manager.set_language('th')
            st.rerun()
    with col2:
        if st.button("🇺🇸 English", key="lang_en"):
            localization_manager.set_language('en')
            st.rerun()
