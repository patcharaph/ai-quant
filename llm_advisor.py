"""
LLM-powered Investment Advisor using OpenRouter API
"""

import os
import requests
import json
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class LLMAdvisor:
    """LLM-powered investment advisor for human-readable recommendations"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.model = os.getenv('OPENROUTER_MODEL', 'openrouter/auto')
        self.base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '500'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        
        # Available models with auto-selection
        self.available_models = {
            'Auto (Best)': 'openrouter/auto',
            'GPT-4o Mini': 'openai/gpt-4o-mini',
            'GPT-3.5 Turbo': 'openai/gpt-3.5-turbo',
            'Claude 3 Haiku': 'anthropic/claude-3-haiku',
            'Claude 3 Sonnet': 'anthropic/claude-3-sonnet',
            'Gemini Pro': 'google/gemini-pro',
            'Llama 3.1 8B': 'meta-llama/llama-3.1-8b-instruct',
            'Mixtral 8x7B': 'mistralai/mixtral-8x7b-instruct',
            'Qwen 2.5 7B': 'qwen/qwen-2.5-7b-instruct'
        }
        
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.api_key is not None and self.api_key != 'your_openrouter_api_key_here'
    
    def generate_human_advice(self, 
                            symbol: str,
                            predicted_return: float,
                            target_return: float,
                            hit_probability: float,
                            expected_return: float,
                            model_performance: Dict[str, float],
                            backtest_metrics: Dict[str, float],
                            market_context: str = "general",
                            selected_model: str = None) -> Dict[str, str]:
        """
        Generate human-readable investment advice using LLM
        
        Args:
            symbol: Stock symbol
            predicted_return: Predicted return percentage
            target_return: Target return percentage
            hit_probability: Probability of hitting target
            expected_return: Expected return percentage
            model_performance: Model performance metrics
            backtest_metrics: Backtesting results
            market_context: Market context description
            
        Returns:
            dict: Human-readable advice in Thai and English
        """
        if not self.is_available():
            return self._fallback_advice(predicted_return, target_return, hit_probability)
        
        try:
            # Use selected model or default
            model_to_use = selected_model if selected_model else self.model
            
            # Prepare context for LLM
            context = self._prepare_context(
                symbol, predicted_return, target_return, hit_probability,
                expected_return, model_performance, backtest_metrics, market_context
            )
            
            # Generate Thai advice
            thai_advice = self._call_llm(context, language="thai", model=model_to_use)
            
            # Generate English advice
            english_advice = self._call_llm(context, language="english", model=model_to_use)
            
            return {
                'thai': thai_advice,
                'english': english_advice,
                'llm_enhanced': True,
                'model_used': model_to_use
            }
            
        except Exception as e:
            print(f"LLM API Error: {e}")
            return self._fallback_advice(predicted_return, target_return, hit_probability)
    
    def _prepare_context(self, symbol, predicted_return, target_return, hit_probability,
                        expected_return, model_performance, backtest_metrics, market_context):
        """Prepare context for LLM"""
        return {
            'symbol': symbol,
            'predicted_return': predicted_return,
            'target_return': target_return,
            'hit_probability': hit_probability,
            'expected_return': expected_return,
            'model_performance': model_performance,
            'backtest_metrics': backtest_metrics,
            'market_context': market_context
        }
    
    def _call_llm(self, context: Dict[str, Any], language: str = "english", model: str = None) -> str:
        """Call OpenRouter API"""
        
        # Create prompt based on language
        if language == "thai":
            prompt = self._create_thai_prompt(context)
        else:
            prompt = self._create_english_prompt(context)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use selected model or default
        model_to_use = model if model else self.model
        
        data = {
            "model": model_to_use,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert investment advisor. Provide clear, educational, and balanced investment advice. Always include risk warnings and disclaimers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def _create_thai_prompt(self, context: Dict[str, Any]) -> str:
        """Create Thai prompt for LLM"""
        return f"""
คุณเป็นที่ปรึกษาการลงทุนผู้เชี่ยวชาญ กรุณาวิเคราะห์ข้อมูลการทำนายหุ้นต่อไปนี้และให้คำแนะนำเป็นภาษาไทยที่เข้าใจง่าย:

**ข้อมูลหุ้น:**
- สัญลักษณ์: {context['symbol']}
- ผลตอบแทนที่คาดการณ์: {context['predicted_return']:.2f}%
- เป้าหมายผลตอบแทน: {context['target_return']:.2f}%
- ความน่าจะเป็นถึงเป้า: {context['hit_probability']:.1%}
- ผลตอบแทนที่คาดหวัง: {context['expected_return']:.2f}%

**ประสิทธิภาพโมเดล:**
- RMSE: {context['model_performance'].get('RMSE', 0):.4f}
- MAE: {context['model_performance'].get('MAE', 0):.4f}
- R²: {context['model_performance'].get('R2', 0):.4f}

**ผลการ Backtest:**
- CAGR: {context['backtest_metrics'].get('CAGR', 0):.2f}%
- Sharpe Ratio: {context['backtest_metrics'].get('Sharpe', 0):.2f}
- Max Drawdown: {context['backtest_metrics'].get('Max_Drawdown', 0):.2f}%
- Win Rate: {context['backtest_metrics'].get('Win_Rate', 0):.1f}%

กรุณาให้คำแนะนำที่:
1. อธิบายผลการทำนายในภาษาที่เข้าใจง่าย
2. วิเคราะห์ความเสี่ยงและโอกาส
3. ให้คำแนะนำการลงทุนที่สมดุล
4. เน้นย้ำว่าเป็นการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน
5. ใช้โทนเสียงที่เป็นมิตรและให้ความรู้

ตอบเป็นภาษาไทยเท่านั้น และใช้ emoji เพื่อให้อ่านง่าย
"""
    
    def _create_english_prompt(self, context: Dict[str, Any]) -> str:
        """Create English prompt for LLM"""
        return f"""
You are an expert investment advisor. Please analyze the following stock prediction data and provide clear, educational investment advice in English:

**Stock Data:**
- Symbol: {context['symbol']}
- Predicted Return: {context['predicted_return']:.2f}%
- Target Return: {context['target_return']:.2f}%
- Hit Probability: {context['hit_probability']:.1%}
- Expected Return: {context['expected_return']:.2f}%

**Model Performance:**
- RMSE: {context['model_performance'].get('RMSE', 0):.4f}
- MAE: {context['model_performance'].get('MAE', 0):.4f}
- R²: {context['model_performance'].get('R2', 0):.4f}

**Backtest Results:**
- CAGR: {context['backtest_metrics'].get('CAGR', 0):.2f}%
- Sharpe Ratio: {context['backtest_metrics'].get('Sharpe', 0):.2f}
- Max Drawdown: {context['backtest_metrics'].get('Max_Drawdown', 0):.2f}%
- Win Rate: {context['backtest_metrics'].get('Win_Rate', 0):.1f}%

Please provide advice that:
1. Explains the prediction results in simple language
2. Analyzes risks and opportunities
3. Gives balanced investment recommendations
4. Emphasizes this is for educational purposes only, not investment advice
5. Uses a friendly and educational tone

Respond in English only and use emojis to make it easy to read.
"""
    
    def _fallback_advice(self, predicted_return: float, target_return: float, hit_probability: float) -> Dict[str, str]:
        """Fallback advice when LLM is not available"""
        return {
            'thai': f"""
🤖 **คำแนะนำจากระบบ AI (ไม่ใช้ LLM)**

📊 **ผลการวิเคราะห์:**
• ผลตอบแทนที่คาดการณ์: {predicted_return:.2f}%
• เป้าหมาย: {target_return:.2f}%
• ความน่าจะเป็นถึงเป้า: {hit_probability:.1%}

💡 **คำแนะนำ:**
{'✅ ดูเหมือนจะมีโอกาสดี' if hit_probability > 0.6 else '⚠️ ควรระมัดระวัง' if hit_probability > 0.4 else '❌ ความเสี่ยงสูง'}

⚠️ **หมายเหตุ**: ข้อมูลนี้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน
""",
            'english': f"""
🤖 **AI System Recommendation (No LLM)**

📊 **Analysis Results:**
• Predicted Return: {predicted_return:.2f}%
• Target: {target_return:.2f}%
• Hit Probability: {hit_probability:.1%}

💡 **Recommendation:**
{'✅ Looks promising' if hit_probability > 0.6 else '⚠️ Be cautious' if hit_probability > 0.4 else '❌ High risk'}

⚠️ **Disclaimer**: This information is for educational purposes only, not investment advice.
""",
            'llm_enhanced': False
        }
    
    def generate_market_insight(self, symbol: str, recent_performance: Dict[str, Any]) -> str:
        """Generate market insight for a specific symbol"""
        if not self.is_available():
            return "LLM service not available for market insights."
        
        try:
            prompt = f"""
Provide a brief market insight for {symbol} based on recent performance:
- Recent price movement: {recent_performance.get('price_change', 0):.2f}%
- Volume trend: {recent_performance.get('volume_trend', 'stable')}
- Volatility: {recent_performance.get('volatility', 0):.2f}%

Keep it concise and educational. Respond in English.
"""
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a market analyst. Provide brief, educational market insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return "Unable to generate market insight at this time."
                
        except Exception as e:
            return f"Error generating insight: {str(e)}"
