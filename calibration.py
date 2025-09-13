"""
Probability Calibration and Uncertainty Quantification

This module provides tools for calibrating prediction intervals and measuring
the reliability of uncertainty estimates in time series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CalibrationMetrics:
    """Calculate calibration metrics for prediction intervals"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_picp(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray, 
                      actual_values: np.ndarray) -> float:
        """
        Calculate Prediction Interval Coverage Probability (PICP)
        
        PICP measures the percentage of actual values that fall within the prediction interval.
        For a well-calibrated 95% interval, PICP should be close to 0.95.
        
        Args:
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals  
            actual_values: Actual observed values
            
        Returns:
            float: PICP value (0-1)
        """
        if len(lower_bounds) != len(upper_bounds) or len(lower_bounds) != len(actual_values):
            raise ValueError("All arrays must have the same length")
        
        # Count how many actual values fall within the prediction interval
        within_interval = (actual_values >= lower_bounds) & (actual_values <= upper_bounds)
        picp = within_interval.mean()
        
        logger.info(f"📊 PICP: {picp:.3f} ({picp*100:.1f}% of actual values within prediction interval)")
        
        return picp
    
    def calculate_pinaw(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray, 
                       actual_values: np.ndarray) -> float:
        """
        Calculate Prediction Interval Normalized Average Width (PINAW)
        
        PINAW measures the average width of prediction intervals normalized by the range of actual values.
        Lower PINAW indicates more precise (narrower) intervals.
        
        Args:
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            actual_values: Actual observed values
            
        Returns:
            float: PINAW value (0-1)
        """
        if len(lower_bounds) != len(upper_bounds) or len(lower_bounds) != len(actual_values):
            raise ValueError("All arrays must have the same length")
        
        # Calculate interval widths
        interval_widths = upper_bounds - lower_bounds
        
        # Normalize by the range of actual values
        actual_range = actual_values.max() - actual_values.min()
        if actual_range == 0:
            return 0.0
        
        pinaw = interval_widths.mean() / actual_range
        
        logger.info(f"📏 PINAW: {pinaw:.3f} (normalized average interval width)")
        
        return pinaw
    
    def calculate_mpiw(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> float:
        """
        Calculate Mean Prediction Interval Width (MPIW)
        
        MPIW measures the average width of prediction intervals.
        
        Args:
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            
        Returns:
            float: MPIW value
        """
        if len(lower_bounds) != len(upper_bounds):
            raise ValueError("Lower and upper bounds must have the same length")
        
        interval_widths = upper_bounds - lower_bounds
        mpiw = interval_widths.mean()
        
        logger.info(f"📐 MPIW: {mpiw:.3f} (mean prediction interval width)")
        
        return mpiw
    
    def calculate_reliability_score(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                                   actual_values: np.ndarray, target_coverage: float = 0.95) -> Dict:
        """
        Calculate reliability score comparing actual coverage to target coverage
        
        Args:
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            actual_values: Actual observed values
            target_coverage: Target coverage probability (e.g., 0.95 for 95% intervals)
            
        Returns:
            dict: Reliability metrics
        """
        picp = self.calculate_picp(lower_bounds, upper_bounds, actual_values)
        pinaw = self.calculate_pinaw(lower_bounds, upper_bounds, actual_values)
        mpiw = self.calculate_mpiw(lower_bounds, upper_bounds)
        
        # Calculate reliability score (penalty for both under-coverage and over-coverage)
        coverage_error = abs(picp - target_coverage)
        reliability_score = 1.0 - coverage_error - pinaw  # Higher is better
        
        # Calculate calibration quality
        if coverage_error < 0.05:  # Within 5% of target
            calibration_quality = "Excellent"
        elif coverage_error < 0.10:  # Within 10% of target
            calibration_quality = "Good"
        elif coverage_error < 0.20:  # Within 20% of target
            calibration_quality = "Fair"
        else:
            calibration_quality = "Poor"
        
        metrics = {
            'picp': picp,
            'pinaw': pinaw,
            'mpiw': mpiw,
            'target_coverage': target_coverage,
            'coverage_error': coverage_error,
            'reliability_score': reliability_score,
            'calibration_quality': calibration_quality,
            'n_samples': len(actual_values)
        }
        
        logger.info(f"🎯 Reliability Score: {reliability_score:.3f}")
        logger.info(f"📈 Calibration Quality: {calibration_quality}")
        
        return metrics
    
    def plot_calibration_curve(self, predicted_probs: np.ndarray, actual_binary: np.ndarray,
                              n_bins: int = 10, title: str = "Calibration Curve") -> plt.Figure:
        """
        Plot calibration curve for binary classification probabilities
        
        Args:
            predicted_probs: Predicted probabilities (0-1)
            actual_binary: Actual binary outcomes (0 or 1)
            n_bins: Number of bins for calibration curve
            title: Plot title
            
        Returns:
            matplotlib.Figure: Calibration curve plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actual_binary, predicted_probs, n_bins=n_bins
        )
        
        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f"Model (Brier: {brier_score_loss(actual_binary, predicted_probs):.3f})")
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_prediction_intervals(self, timestamps: np.ndarray, actual_values: np.ndarray,
                                predictions: np.ndarray, lower_bounds: np.ndarray,
                                upper_bounds: np.ndarray, title: str = "Prediction Intervals") -> plt.Figure:
        """
        Plot prediction intervals with actual values
        
        Args:
            timestamps: Time indices
            actual_values: Actual observed values
            predictions: Point predictions
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            title: Plot title
            
        Returns:
            matplotlib.Figure: Prediction interval plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot prediction interval
        ax.fill_between(timestamps, lower_bounds, upper_bounds, 
                       alpha=0.3, color='blue', label='Prediction Interval')
        
        # Plot actual values
        ax.plot(timestamps, actual_values, 'o-', color='red', 
               markersize=3, label='Actual', alpha=0.7)
        
        # Plot predictions
        ax.plot(timestamps, predictions, '--', color='blue', 
               linewidth=2, label='Prediction')
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def calculate_quantile_scores(self, predictions: np.ndarray, actual_values: np.ndarray,
                                 quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> Dict:
        """
        Calculate quantile scores for different prediction quantiles
        
        Args:
            predictions: Model predictions
            actual_values: Actual observed values
            quantiles: List of quantiles to evaluate
            
        Returns:
            dict: Quantile scores
        """
        quantile_scores = {}
        
        for q in quantiles:
            # Calculate quantile score (lower is better)
            errors = actual_values - predictions
            score = np.mean(np.maximum((q - 1) * errors, q * errors))
            quantile_scores[f'quantile_{q:.2f}'] = score
            
        logger.info(f"📊 Quantile scores: {quantile_scores}")
        
        return quantile_scores


class UncertaintyQuantifier:
    """Quantify uncertainty in time series predictions"""
    
    def __init__(self):
        self.calibration_metrics = CalibrationMetrics()
    
    def monte_carlo_dropout(self, model, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate uncertainty estimates using Monte Carlo Dropout
        
        Args:
            model: Trained model with dropout layers
            X: Input features
            n_samples: Number of Monte Carlo samples
            
        Returns:
            tuple: (mean_predictions, lower_bounds, upper_bounds)
        """
        predictions = []
        
        for _ in range(n_samples):
            # Enable dropout during inference
            pred = model.predict(X, training=True)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate prediction intervals (95% confidence)
        lower_bounds = mean_pred - 1.96 * std_pred
        upper_bounds = mean_pred + 1.96 * std_pred
        
        return mean_pred, lower_bounds, upper_bounds
    
    def bootstrap_uncertainty(self, model, X: np.ndarray, y: np.ndarray, 
                            n_bootstrap: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate uncertainty estimates using Bootstrap
        
        Args:
            model: Model class to train
            X: Training features
            y: Training targets
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            tuple: (mean_predictions, lower_bounds, upper_bounds)
        """
        predictions = []
        n_samples = len(X)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model on bootstrap sample
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_boot, y_boot)
            
            # Make predictions
            pred = model_copy.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        lower_bounds = np.percentile(predictions, 2.5, axis=0)
        upper_bounds = np.percentile(predictions, 97.5, axis=0)
        
        return mean_pred, lower_bounds, upper_bounds
    
    def evaluate_uncertainty(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                           actual_values: np.ndarray, target_coverage: float = 0.95) -> Dict:
        """
        Comprehensive evaluation of uncertainty estimates
        
        Args:
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            actual_values: Actual observed values
            target_coverage: Target coverage probability
            
        Returns:
            dict: Comprehensive uncertainty evaluation metrics
        """
        # Calculate calibration metrics
        reliability_metrics = self.calibration_metrics.calculate_reliability_score(
            lower_bounds, upper_bounds, actual_values, target_coverage
        )
        
        # Calculate additional metrics
        interval_width = upper_bounds - lower_bounds
        width_std = np.std(interval_width)
        width_cv = width_std / np.mean(interval_width) if np.mean(interval_width) > 0 else 0
        
        # Calculate sharpness (inverse of average width)
        sharpness = 1.0 / np.mean(interval_width) if np.mean(interval_width) > 0 else 0
        
        evaluation = {
            **reliability_metrics,
            'interval_width_std': width_std,
            'interval_width_cv': width_cv,
            'sharpness': sharpness,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"🎯 Uncertainty Evaluation Complete:")
        logger.info(f"   PICP: {reliability_metrics['picp']:.3f}")
        logger.info(f"   PINAW: {reliability_metrics['pinaw']:.3f}")
        logger.info(f"   Reliability Score: {reliability_metrics['reliability_score']:.3f}")
        logger.info(f"   Calibration Quality: {reliability_metrics['calibration_quality']}")
        
        return evaluation


def create_calibration_report(predictions: np.ndarray, actual_values: np.ndarray,
                            lower_bounds: Optional[np.ndarray] = None,
                            upper_bounds: Optional[np.ndarray] = None,
                            target_coverage: float = 0.95) -> Dict:
    """
    Create a comprehensive calibration report
    
    Args:
        predictions: Model predictions
        actual_values: Actual observed values
        lower_bounds: Lower bounds of prediction intervals (optional)
        upper_bounds: Upper bounds of prediction intervals (optional)
        target_coverage: Target coverage probability
        
    Returns:
        dict: Comprehensive calibration report
    """
    quantifier = UncertaintyQuantifier()
    
    report = {
        'basic_metrics': {
            'mae': np.mean(np.abs(actual_values - predictions)),
            'rmse': np.sqrt(np.mean((actual_values - predictions) ** 2)),
            'mape': np.mean(np.abs((actual_values - predictions) / actual_values)) * 100,
            'r2': 1 - np.sum((actual_values - predictions) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
        }
    }
    
    # Add uncertainty evaluation if intervals are provided
    if lower_bounds is not None and upper_bounds is not None:
        report['uncertainty_evaluation'] = quantifier.evaluate_uncertainty(
            lower_bounds, upper_bounds, actual_values, target_coverage
        )
    
    return report
