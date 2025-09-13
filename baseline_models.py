"""
Baseline Models for Time Series Forecasting

This module provides simple baseline models for comparison with more complex
neural network models. These baselines help establish performance benchmarks.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️  statsmodels not available. ARIMA and ExponentialSmoothing models disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  prophet not available. Prophet model disabled.")


class NaiveModel:
    """Naive baseline: predict last value"""
    
    def __init__(self):
        self.last_value = None
        self.name = "Naive"
    
    def fit(self, X, y):
        """Fit the naive model"""
        self.last_value = y[-1] if len(y) > 0 else 0
        return self
    
    def predict(self, X):
        """Predict using last value"""
        return np.full(len(X), self.last_value)
    
    def get_params(self):
        return {'last_value': self.last_value}


class LinearBaseline:
    """Linear regression baseline"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.name = "Linear"
    
    def fit(self, X, y):
        """Fit linear regression"""
        # Reshape X if needed (for time series data)
        if len(X.shape) == 3:  # (samples, timesteps, features)
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        self.model.fit(X_reshaped, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        if len(X.shape) == 3:  # (samples, timesteps, features)
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        return self.model.predict(X_reshaped)
    
    def get_params(self):
        return {
            'coef_': self.model.coef_,
            'intercept_': self.model.intercept_,
            'score_': self.model.score
        }


class ARIMABaseline:
    """ARIMA baseline model"""
    
    def __init__(self, order=(1, 1, 1)):
        if not ARIMA_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA model")
        
        self.order = order
        self.model = None
        self.name = f"ARIMA{order}"
    
    def fit(self, X, y):
        """Fit ARIMA model"""
        try:
            self.model = ARIMA(y, order=self.order)
            self.fitted_model = self.model.fit()
            return self
        except Exception as e:
            print(f"⚠️  ARIMA fitting failed: {e}")
            # Fallback to naive model
            self.model = NaiveModel()
            return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        if hasattr(self, 'fitted_model'):
            try:
                # Get number of steps to forecast
                n_steps = len(X)
                forecast = self.fitted_model.forecast(steps=n_steps)
                return forecast
            except Exception as e:
                print(f"⚠️  ARIMA prediction failed: {e}")
                return np.full(len(X), self.fitted_model.fittedvalues[-1])
        else:
            return self.model.predict(X)
    
    def get_params(self):
        if hasattr(self, 'fitted_model'):
            return {
                'order': self.order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'params': self.fitted_model.params
            }
        else:
            return {'order': self.order, 'fallback': True}


class ProphetBaseline:
    """Prophet baseline model"""
    
    def __init__(self, **kwargs):
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet model")
        
        self.prophet_kwargs = kwargs
        self.model = None
        self.name = "Prophet"
    
    def fit(self, X, y):
        """Fit Prophet model"""
        try:
            # Create DataFrame for Prophet
            dates = pd.date_range(start='2020-01-01', periods=len(y), freq='D')
            df = pd.DataFrame({
                'ds': dates,
                'y': y
            })
            
            self.model = Prophet(**self.prophet_kwargs)
            self.model.fit(df)
            return self
        except Exception as e:
            print(f"⚠️  Prophet fitting failed: {e}")
            # Fallback to naive model
            self.model = NaiveModel()
            return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        if hasattr(self, 'model') and hasattr(self.model, 'predict'):
            try:
                # Create future DataFrame
                n_steps = len(X)
                future_dates = pd.date_range(start='2020-01-01', periods=len(X) + n_steps, freq='D')
                future_df = pd.DataFrame({'ds': future_dates[-n_steps:]})
                
                forecast = self.model.predict(future_df)
                return forecast['yhat'].values
            except Exception as e:
                print(f"⚠️  Prophet prediction failed: {e}")
                return np.full(len(X), 0)
        else:
            return self.model.predict(X)
    
    def get_params(self):
        if hasattr(self, 'model') and hasattr(self.model, 'params'):
            return {
                'prophet_kwargs': self.prophet_kwargs,
                'params': self.model.params
            }
        else:
            return {'prophet_kwargs': self.prophet_kwargs, 'fallback': True}


class MovingAverageBaseline:
    """Moving average baseline"""
    
    def __init__(self, window=5):
        self.window = window
        self.name = f"MA{window}"
        self.history = []
    
    def fit(self, X, y):
        """Fit moving average model"""
        self.history = y.tolist()
        return self
    
    def predict(self, X):
        """Make predictions using moving average"""
        predictions = []
        
        for i in range(len(X)):
            if len(self.history) >= self.window:
                ma_value = np.mean(self.history[-self.window:])
            else:
                ma_value = np.mean(self.history) if self.history else 0
            
            predictions.append(ma_value)
            # Update history (simulating online prediction)
            if i < len(self.history):
                self.history.append(self.history[i])
        
        return np.array(predictions)
    
    def get_params(self):
        return {'window': self.window}


class BaselineModelSelector:
    """Select and evaluate baseline models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model):
        """Add a baseline model"""
        self.models[name] = model
    
    def evaluate_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Evaluate all baseline models
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            dict: Evaluation results for all models
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"🔄 Evaluating {name} baseline model...")
            
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, train_pred, f"{name}_train")
                val_metrics = self._calculate_metrics(y_val, val_pred, f"{name}_val")
                test_metrics = self._calculate_metrics(y_test, test_pred, f"{name}_test")
                
                results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'predictions': {
                        'train': train_pred,
                        'val': val_pred,
                        'test': test_pred
                    }
                }
                
                print(f"✅ {name} - Val RMSE: {val_metrics['rmse']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def _calculate_metrics(self, y_true, y_pred, prefix=""):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': 1 - mse / np.var(y_true) if np.var(y_true) > 0 else 0
        }
    
    def get_best_model(self, metric='rmse', dataset='val'):
        """Get the best performing model based on specified metric"""
        if not self.results:
            return None
        
        best_model = None
        best_score = float('inf')
        
        for name, result in self.results.items():
            if 'error' in result:
                continue
            
            score = result[f'{dataset}_metrics'][metric]
            if score < best_score:
                best_score = score
                best_model = name
        
        return best_model, best_score
    
    def create_baseline_comparison_report(self) -> pd.DataFrame:
        """Create a comparison report of all baseline models"""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, result in self.results.items():
            if 'error' in result:
                continue
            
            row = {'Model': name}
            
            # Add validation metrics
            val_metrics = result['val_metrics']
            for metric, value in val_metrics.items():
                row[f'Val_{metric.upper()}'] = value
            
            # Add test metrics
            test_metrics = result['test_metrics']
            for metric, value in test_metrics.items():
                row[f'Test_{metric.upper()}'] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values('Val_RMSE')


def create_default_baseline_models():
    """Create a set of default baseline models"""
    selector = BaselineModelSelector()
    
    # Add basic baselines
    selector.add_model('Naive', NaiveModel())
    selector.add_model('Linear', LinearBaseline())
    selector.add_model('MA5', MovingAverageBaseline(window=5))
    selector.add_model('MA10', MovingAverageBaseline(window=10))
    
    # Add ARIMA if available
    if ARIMA_AVAILABLE:
        selector.add_model('ARIMA(1,1,1)', ARIMABaseline(order=(1, 1, 1)))
        selector.add_model('ARIMA(2,1,1)', ARIMABaseline(order=(2, 1, 1)))
    
    # Add Prophet if available
    if PROPHET_AVAILABLE:
        selector.add_model('Prophet', ProphetBaseline())
    
    return selector


def run_baseline_comparison(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Run comprehensive baseline model comparison
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data  
        X_test, y_test: Test data
        
    Returns:
        tuple: (results_dict, comparison_dataframe)
    """
    # Create baseline models
    selector = create_default_baseline_models()
    
    # Evaluate all models
    results = selector.evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Create comparison report
    comparison_df = selector.create_baseline_comparison_report()
    
    # Print summary
    print("\n📊 Baseline Model Comparison Summary:")
    print("=" * 50)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Get best model
    best_model_name, best_score = selector.get_best_model('rmse', 'val')
    print(f"\n🏆 Best Baseline Model: {best_model_name} (Val RMSE: {best_score:.4f})")
    
    return results, comparison_df
