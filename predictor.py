"""
Prediction and Risk Calculation Module
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class Predictor:
    """Handles prediction and risk calculations"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
    def forecast(self, model, X_latest, horizon_days=1):
        """
        Make forecast for the given horizon
        
        Args:
            model: Trained model (LSTM or Transformer)
            X_latest: Latest input sequence
            horizon_days: Prediction horizon
            
        Returns:
            dict: Forecast results with uncertainty
        """
        # Single prediction
        y_hat = model.predict(X_latest.reshape(1, -1, X_latest.shape[-1]))[0]
        
        # For now, return point estimate
        # In a more sophisticated implementation, we could use:
        # - Monte Carlo dropout
        # - Ensemble methods
        # - Bootstrap sampling
        
        return {
            'y_hat': y_hat,
            'forecast_type': 'point_estimate',
            'horizon_days': horizon_days
        }
    
    def calculate_predicted_return(self, y_hat, current_price, target_type='price'):
        """
        Calculate predicted return percentage
        
        Args:
            y_hat: Predicted value
            current_price: Current stock price
            target_type: 'price' or 'return'
            
        Returns:
            float: Predicted return percentage
        """
        if target_type == 'price':
            # y_hat is predicted price, calculate return
            return ((y_hat - current_price) / current_price) * 100
        else:
            # y_hat is already return percentage
            return y_hat
    
    def calculate_prediction_interval(self, residuals, confidence_level=0.95):
        """
        Calculate prediction interval based on residuals
        
        Args:
            residuals: Model residuals from validation/test set
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            dict: Prediction interval statistics
        """
        # Calculate residual statistics
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Prediction interval bounds
        pi_low = residual_mean - z_score * residual_std
        pi_high = residual_mean + z_score * residual_std
        
        return {
            'residual_std': residual_std,
            'residual_mean': residual_mean,
            'pi_low': pi_low,
            'pi_high': pi_high,
            'confidence_level': confidence_level,
            'z_score': z_score
        }
    
    def calculate_hit_probability(self, predicted_return, target_return, residual_std, method='normal'):
        """
        Calculate probability of hitting target return
        
        Args:
            predicted_return: Predicted return percentage
            target_return: Target return percentage
            residual_std: Standard deviation of residuals
            method: 'normal' or 'bootstrap'
            
        Returns:
            float: Probability of hitting target (0-1)
        """
        if method == 'normal':
            # Normal approximation
            if residual_std == 0:
                return 1.0 if predicted_return >= target_return else 0.0
            
            z_score = (predicted_return - target_return) / residual_std
            hit_prob = 1 - stats.norm.cdf(z_score)
            return max(0, min(1, hit_prob))  # Clamp between 0 and 1
            
        elif method == 'bootstrap':
            # Bootstrap method (would need actual residuals)
            # For now, return normal approximation
            return self.calculate_hit_probability(predicted_return, target_return, residual_std, 'normal')
        
        else:
            raise ValueError("Method must be 'normal' or 'bootstrap'")
    
    def calculate_expected_return(self, predicted_return, hit_probability, target_return):
        """
        Calculate expected return considering uncertainty
        
        Args:
            predicted_return: Predicted return percentage
            hit_probability: Probability of hitting target
            target_return: Target return percentage
            
        Returns:
            dict: Expected return analysis
        """
        # Simple expected return calculation
        # In practice, this could be more sophisticated
        expected_return = predicted_return * hit_probability + target_return * (1 - hit_probability)
        
        return {
            'expected_return': expected_return,
            'predicted_return': predicted_return,
            'hit_probability': hit_probability,
            'target_return': target_return,
            'risk_adjusted_return': expected_return / (1 + abs(predicted_return - target_return) / 100)
        }

class RiskCalculator:
    """Risk calculation utilities"""
    
    @staticmethod
    def calculate_volatility(returns, window=30):
        """Calculate rolling volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252) * 100
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(prices):
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min() * 100
    
    @staticmethod
    def calculate_value_at_risk(returns, confidence_level=0.05):
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns, confidence_level=0.05):
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = RiskCalculator.calculate_value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()

class AdvisoryGenerator:
    """Generate investment advisory based on predictions"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
    def generate_advisory(self, predicted_return, target_return, hit_probability, 
                         expected_return, risk_level='medium'):
        """
        Generate investment advisory in Thai and English
        
        Args:
            predicted_return: Predicted return percentage
            target_return: Target return percentage
            hit_probability: Probability of hitting target
            expected_return: Expected return percentage
            risk_level: Risk level ('low', 'medium', 'high')
            
        Returns:
            dict: Advisory in both languages
        """
        # Convert to float to avoid numpy formatting issues
        predicted_return = float(predicted_return)
        target_return = float(target_return)
        hit_probability = float(hit_probability)
        expected_return = float(expected_return)
        # Determine advisory level
        if hit_probability >= 0.6 and expected_return >= target_return * 0.8:
            level = 'positive'
        elif 0.4 <= hit_probability < 0.6:
            level = 'neutral'
        else:
            level = 'negative'
        
        # Generate Thai advisory
        th_advisory = self._generate_thai_advisory(
            predicted_return, target_return, hit_probability, 
            expected_return, level, risk_level
        )
        
        # Generate English advisory
        en_advisory = self._generate_english_advisory(
            predicted_return, target_return, hit_probability, 
            expected_return, level, risk_level
        )
        
        return {
            'thai': th_advisory,
            'english': en_advisory,
            'level': level,
            'confidence': hit_probability,
            'risk_level': risk_level
        }
    
    def _generate_thai_advisory(self, pred_ret, target_ret, hit_prob, exp_ret, level, risk_level):
        """Generate Thai advisory text"""
        hit_prob_pct = hit_prob * 100
        
        if level == 'positive':
            return f"""
            📈 **การวิเคราะห์เชิงบวก**
            
            • **ผลตอบแทนที่คาดการณ์**: {pred_ret:.2f}%
            • **เป้าหมาย**: {target_ret:.2f}%
            • **ความน่าจะเป็นถึงเป้า**: {hit_prob_pct:.1f}%
            • **ผลตอบแทนที่คาดหวัง**: {exp_ret:.2f}%
            
            **คำแนะนำ**: ตามการวิเคราะห์ มีโอกาสสูงที่จะบรรลุเป้าหมายผลตอบแทน 
            อย่างไรก็ตาม การลงทุนมีความเสี่ยง โปรดพิจารณาให้รอบคอบ
            
            ⚠️ **หมายเหตุ**: ข้อมูลนี้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน
            """
        elif level == 'neutral':
            return f"""
            ⚖️ **การวิเคราะห์กลาง**
            
            • **ผลตอบแทนที่คาดการณ์**: {pred_ret:.2f}%
            • **เป้าหมาย**: {target_ret:.2f}%
            • **ความน่าจะเป็นถึงเป้า**: {hit_prob_pct:.1f}%
            • **ผลตอบแทนที่คาดหวัง**: {exp_ret:.2f}%
            
            **คำแนะนำ**: ผลการทำนายไม่ชัดเจน แนะนำให้ปรับเป้าหมายผลตอบแทน 
            หรือขยายระยะเวลาการลงทุนเพื่อลดความเสี่ยง
            
            ⚠️ **หมายเหตุ**: ข้อมูลนี้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน
            """
        else:  # negative
            return f"""
            📉 **การวิเคราะห์เชิงลบ**
            
            • **ผลตอบแทนที่คาดการณ์**: {pred_ret:.2f}%
            • **เป้าหมาย**: {target_ret:.2f}%
            • **ความน่าจะเป็นถึงเป้า**: {hit_prob_pct:.1f}%
            • **ผลตอบแทนที่คาดหวัง**: {exp_ret:.2f}%
            
            **คำแนะนำ**: โอกาสบรรลุเป้าหมายต่ำ แนะนำให้ลดเป้าหมายผลตอบแทน 
            หรือเพิ่มระยะเวลาการลงทุน หรือพิจารณาเครื่องมือลงทุนอื่น
            
            ⚠️ **หมายเหตุ**: ข้อมูลนี้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน
            """
    
    def _generate_english_advisory(self, pred_ret, target_ret, hit_prob, exp_ret, level, risk_level):
        """Generate English advisory text"""
        hit_prob_pct = hit_prob * 100
        
        if level == 'positive':
            return f"""
            📈 **Positive Analysis**
            
            • **Predicted Return**: {pred_ret:.2f}%
            • **Target Return**: {target_ret:.2f}%
            • **Hit Probability**: {hit_prob_pct:.1f}%
            • **Expected Return**: {exp_ret:.2f}%
            
            **Recommendation**: Based on analysis, there's a high probability of achieving 
            the target return. However, investing involves risks, please consider carefully.
            
            ⚠️ **Disclaimer**: This information is for educational purposes only, not investment advice.
            """
        elif level == 'neutral':
            return f"""
            ⚖️ **Neutral Analysis**
            
            • **Predicted Return**: {pred_ret:.2f}%
            • **Target Return**: {target_ret:.2f}%
            • **Hit Probability**: {hit_prob_pct:.1f}%
            • **Expected Return**: {exp_ret:.2f}%
            
            **Recommendation**: Prediction results are unclear. Consider adjusting target return 
            or extending investment horizon to reduce risk.
            
            ⚠️ **Disclaimer**: This information is for educational purposes only, not investment advice.
            """
        else:  # negative
            return f"""
            📉 **Negative Analysis**
            
            • **Predicted Return**: {pred_ret:.2f}%
            • **Target Return**: {target_ret:.2f}%
            • **Hit Probability**: {hit_prob_pct:.1f}%
            • **Expected Return**: {exp_ret:.2f}%
            
            **Recommendation**: Low probability of achieving target. Consider reducing target return, 
            extending investment horizon, or exploring other investment instruments.
            
            ⚠️ **Disclaimer**: This information is for educational purposes only, not investment advice.
            """
