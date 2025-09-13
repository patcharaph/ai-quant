"""
LLM-powered Investment Advisor using OpenRouter API
"""

import os
import requests
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

class LLMAdvisor:
    """LLM-powered investment advisor for human-readable recommendations"""
    
    def __init__(self, config=None):
        self.config = config or {}
        from env_manager import get_env_config
        env_config = get_env_config()
        
        self.api_key = env_config.OPENROUTER_API_KEY
        self.model = env_config.OPENROUTER_MODEL
        self.base_url = env_config.OPENROUTER_BASE_URL
        self.max_tokens = env_config.MAX_TOKENS
        self.temperature = env_config.TEMPERATURE
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.timeout = 30  # seconds
        
        # Logging configuration
        self.log_dir = Path('llm_logs')
        self.log_dir.mkdir(exist_ok=True)
        
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
        
        # System prompts for different contexts
        self.system_prompts = {
            'educational': self._get_educational_prompt(),
            'analysis': self._get_analysis_prompt(),
            'risk_assessment': self._get_risk_assessment_prompt()
        }
        
    def _get_educational_prompt(self) -> str:
        """Get educational system prompt"""
        return """You are an educational AI assistant specializing in stock market analysis and financial education. Your role is to:

1. **Educational Focus**: Provide educational insights about stock market concepts, not investment advice
2. **Risk Awareness**: Always emphasize the risks and uncertainties in stock investing
3. **No Recommendations**: Never provide specific buy/sell recommendations or investment advice
4. **Educational Context**: Explain market concepts, technical analysis, and risk factors
5. **Disclaimers**: Always include appropriate disclaimers about investment risks

Key Guidelines:
- Use educational language and explain concepts clearly
- Focus on teaching rather than advising
- Emphasize the importance of diversification and risk management
- Mention that past performance doesn't guarantee future results
- Encourage users to consult with financial professionals
- Be objective and balanced in your analysis

Remember: You are an educational tool, not a financial advisor."""

    def _get_analysis_prompt(self) -> str:
        """Get analysis system prompt"""
        return """You are an AI assistant that provides objective analysis of stock market data and predictions. Your role is to:

1. **Objective Analysis**: Provide neutral, data-driven analysis of market information
2. **Context Explanation**: Help users understand what the data means
3. **Risk Assessment**: Highlight potential risks and limitations
4. **Educational Value**: Explain the methodology and assumptions behind predictions
5. **Balanced Perspective**: Present both positive and negative aspects

Key Guidelines:
- Focus on explaining the data and its implications
- Avoid making specific investment recommendations
- Highlight uncertainties and limitations
- Use clear, accessible language
- Provide context for the analysis
- Encourage critical thinking

Remember: Provide analysis, not advice."""

    def _get_risk_assessment_prompt(self) -> str:
        """Get risk assessment system prompt"""
        return """You are an AI assistant focused on risk assessment and financial education. Your role is to:

1. **Risk Identification**: Identify and explain various types of investment risks
2. **Risk Quantification**: Help users understand risk metrics and their implications
3. **Risk Mitigation**: Suggest general risk management principles (not specific strategies)
4. **Educational Focus**: Teach users about risk concepts and management
5. **Cautionary Guidance**: Emphasize the importance of understanding risks

Key Guidelines:
- Always prioritize risk awareness
- Explain risk metrics in simple terms
- Highlight the potential for loss
- Suggest general risk management principles
- Emphasize the importance of diversification
- Encourage professional consultation

Remember: Focus on risk education, not risk management advice."""

    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.api_key is not None and self.api_key != 'your_openrouter_api_key_here'
    
    def _make_api_request(self, messages: List[Dict], context: str = 'educational') -> Optional[Dict]:
        """Make API request with retry logic and error handling"""
        
        if not self.is_available():
            logger.error("LLM service not available - missing API key")
            return None
        
        # Get appropriate system prompt
        system_prompt = self.system_prompts.get(context, self.system_prompts['educational'])
        
        # Prepare request
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://ai-quant-stock-predictor.streamlit.app',
            'X-Title': 'AI Quant Stock Predictor'
        }
        
        payload = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system_prompt}] + messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'stream': False
        }
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 Making LLM API request (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("✅ LLM API request successful")
                    return result
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    logger.warning(f"⚠️  {error_msg}")
                    last_error = error_msg
                    
            except requests.exceptions.Timeout:
                error_msg = f"Request timeout after {self.timeout} seconds"
                logger.warning(f"⏰ {error_msg}")
                last_error = error_msg
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {str(e)}"
                logger.warning(f"❌ {error_msg}")
                last_error = error_msg
                
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"💥 {error_msg}")
                last_error = error_msg
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        logger.error(f"❌ All retry attempts failed. Last error: {last_error}")
        return None
    
    def _log_interaction(self, symbol: str, context: str, input_data: Dict, 
                        response: Optional[Dict], error: Optional[str] = None):
        """Log LLM interaction for analysis and debugging"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'context': context,
            'model': self.model,
            'input_data': input_data,
            'response': response,
            'error': error,
            'api_key_masked': f"{self.api_key[:8]}..." if self.api_key else None
        }
        
        # Save to file
        log_file = self.log_dir / f"llm_interactions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        logger.info(f"📝 LLM interaction logged to {log_file}")
    
    def _apply_guardrails(self, response_text: str) -> str:
        """Apply guardrails to LLM response"""
        
        # Remove any specific buy/sell recommendations
        forbidden_phrases = [
            'buy this stock',
            'sell this stock',
            'you should buy',
            'you should sell',
            'I recommend buying',
            'I recommend selling',
            'definitely buy',
            'definitely sell',
            'guaranteed return',
            'sure thing',
            'no risk'
        ]
        
        response_lower = response_text.lower()
        for phrase in forbidden_phrases:
            if phrase in response_lower:
                logger.warning(f"⚠️  Guardrail triggered: '{phrase}' detected in response")
                # Replace with educational disclaimer
                response_text = response_text.replace(phrase, f"[Educational analysis - not investment advice]")
        
        # Add educational disclaimer if not present
        if 'not investment advice' not in response_lower and 'educational' not in response_lower:
            response_text += "\n\n⚠️ **Important**: This analysis is for educational purposes only and should not be considered as investment advice. Please consult with a qualified financial advisor before making any investment decisions."
        
        return response_text
    
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
        Generate human-readable educational analysis using LLM with enhanced guardrails
        
        Args:
            symbol: Stock symbol
            predicted_return: Predicted return percentage
            target_return: Target return percentage
            hit_probability: Probability of hitting target
            expected_return: Expected return percentage
            model_performance: Model performance metrics
            backtest_metrics: Backtesting results
            market_context: Market context description
            selected_model: Specific model to use
            context: Analysis context ('educational', 'analysis', 'risk_assessment')
            
        Returns:
            dict: Educational analysis in Thai and English
        """
        if not self.is_available():
            logger.warning("LLM service not available, using fallback")
            return self._fallback_advice(predicted_return, target_return, hit_probability)
        
        try:
            # Prepare input data for logging
            input_data = {
                'symbol': symbol,
                'predicted_return': predicted_return,
                'target_return': target_return,
                'hit_probability': hit_probability,
                'expected_return': expected_return,
                'model_performance': model_performance,
                'backtest_metrics': backtest_metrics,
                'market_context': market_context,
                'selected_model': selected_model,
                'context': context
            }
            
            # Use selected model or default
            model_to_use = selected_model if selected_model else self.model
            
            # Prepare context for LLM
            llm_context = self._prepare_context(
                symbol, predicted_return, target_return, hit_probability,
                expected_return, model_performance, backtest_metrics, market_context
            )
            
            # Generate Thai analysis
            thai_messages = [{
                'role': 'user', 
                'content': f"Please provide an educational analysis in Thai for the following stock data:\n\n{llm_context}\n\nFocus on explaining the concepts and risks, not providing investment advice."
            }]
            
            thai_response = self._make_api_request(thai_messages, context)
            thai_analysis = self._extract_response_text(thai_response) if thai_response else "ไม่สามารถสร้างการวิเคราะห์ได้"
            thai_analysis = self._apply_guardrails(thai_analysis)
            
            # Generate English analysis
            english_messages = [{
                'role': 'user', 
                'content': f"Please provide an educational analysis in English for the following stock data:\n\n{llm_context}\n\nFocus on explaining the concepts and risks, not providing investment advice."
            }]
            
            english_response = self._make_api_request(english_messages, context)
            english_analysis = self._extract_response_text(english_response) if english_response else "Unable to generate analysis"
            english_analysis = self._apply_guardrails(english_analysis)
            
            # Log interactions
            self._log_interaction(symbol, context, input_data, thai_response)
            self._log_interaction(symbol, context, input_data, english_response)
            
            return {
                'thai': thai_analysis,
                'english': english_analysis,
                'llm_enhanced': True,
                'model_used': model_to_use,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LLM API Error: {e}")
            self._log_interaction(symbol, context, input_data, None, str(e))
            return self._fallback_advice(predicted_return, target_return, hit_probability)
    
    def _extract_response_text(self, response: Optional[Dict]) -> str:
        """Extract text from LLM API response"""
        if not response or 'choices' not in response:
            return "No response received"
        
        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting response text: {e}")
            return "Error extracting response"
    
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
