"""
LSTM and Transformer models for time series prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import logging
import json
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for time series prediction"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        # Use fully-qualified path so test patching works
        model = tf.keras.models.Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.config.get('hidden_units', 64),
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(self.config.get('dropout', 0.2)))
        
        # Second LSTM layer (if specified)
        if self.config.get('layers', 2) > 1:
            model.add(LSTM(
                units=self.config.get('hidden_units', 64),
                return_sequences=False
            ))
            model.add(Dropout(self.config.get('dropout', 0.2)))
        
        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.get('learning_rate', 1e-3)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the LSTM model"""
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('patience', 5),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get('epochs', 30),
            batch_size=self.config.get('batch_size', 64),
            callbacks=callbacks,
            verbose=0
        )
        
        return self.model, self.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, verbose=0).flatten()
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            return "Model not built yet"
        return self.model.summary()

class TransformerModel:
    """Transformer model for time series prediction"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """Build Transformer model architecture"""
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Project input to d_model dimension
        d_model = self.config.get('d_model', 64)
        x = Dense(d_model)(inputs)
        
        # Multi-head attention layers
        for _ in range(self.config.get('num_layers', 2)):
            # Self-attention
            attention_output = MultiHeadAttention(
                num_heads=self.config.get('n_heads', 4),
                key_dim=d_model // self.config.get('n_heads', 4),
                dropout=self.config.get('dropout', 0.1)
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed forward
            ff_output = Dense(self.config.get('ff_dim', 128), activation='relu')(x)
            ff_output = Dropout(self.config.get('dropout', 0.1))(ff_output)
            ff_output = Dense(d_model)(ff_output)
            
            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(x + ff_output)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.get('learning_rate', 1e-3)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the Transformer model"""
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('patience', 5),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get('epochs', 30),
            batch_size=self.config.get('batch_size', 64),
            callbacks=callbacks,
            verbose=0
        )
        
        return self.model, self.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, verbose=0).flatten()
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            return "Model not built yet"
        return self.model.summary()

class ModelEvaluator:
    """Model evaluation utilities"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
    
    @staticmethod
    def calculate_direction_accuracy(y_true, y_pred):
        """Calculate direction accuracy"""
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        
        # Remove zero directions
        mask = (true_direction != 0) & (pred_direction != 0)
        if np.sum(mask) == 0:
            return 0.0
        
        accuracy = np.mean(true_direction[mask] == pred_direction[mask])
        return accuracy * 100

class ModelSelector:
    """Model selection logic"""
    
    @staticmethod
    def select_best_model(lstm_metrics, transformer_metrics):
        """
        Select best model based on validation metrics
        
        Selection criteria:
        1. Lowest RMSE on validation set
        2. If RMSE difference < 2%, use MAE
        3. If still tied, use MAPE
        """
        lstm_rmse = lstm_metrics['RMSE']
        trans_rmse = transformer_metrics['RMSE']
        
        # Calculate relative difference
        rmse_diff_pct = abs(lstm_rmse - trans_rmse) / min(lstm_rmse, trans_rmse) * 100
        
        if rmse_diff_pct >= 2.0:
            # RMSE difference is significant
            if lstm_rmse < trans_rmse:
                return 'lstm', f"LSTM has lower RMSE ({lstm_rmse:.4f} vs {trans_rmse:.4f})"
            else:
                return 'transformer', f"Transformer has lower RMSE ({trans_rmse:.4f} vs {lstm_rmse:.4f})"
        else:
            # RMSE difference < 2%, use MAE
            lstm_mae = lstm_metrics['MAE']
            trans_mae = transformer_metrics['MAE']
            
            if lstm_mae < trans_mae:
                return 'lstm', f"RMSE difference < 2%, LSTM has lower MAE ({lstm_mae:.4f} vs {trans_mae:.4f})"
            elif trans_mae < lstm_mae:
                return 'transformer', f"RMSE difference < 2%, Transformer has lower MAE ({trans_mae:.4f} vs {lstm_mae:.4f})"
            else:
                # MAE also tied, use MAPE
                lstm_mape = lstm_metrics['MAPE']
                trans_mape = transformer_metrics['MAPE']
                
                if lstm_mape < trans_mape:
                    return 'lstm', f"RMSE & MAE tied, LSTM has lower MAPE ({lstm_mape:.2f}% vs {trans_mape:.2f}%)"
                else:
                    return 'transformer', f"RMSE & MAE tied, Transformer has lower MAPE ({trans_mape:.2f}% vs {lstm_mape:.2f}%)"


class ModelSelectionLogger:
    """Log model selection decisions and artifacts"""
    
    def __init__(self, log_dir='model_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.selection_log = []
    
    def log_model_selection(self, model_name: str, metrics: dict, 
                           selection_reason: str, hyperparams: dict = None):
        """Log a model selection decision"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'selection_reason': selection_reason,
            'hyperparams': hyperparams or {}
        }
        
        self.selection_log.append(log_entry)
        
        # Save to file
        log_file = self.log_dir / f"model_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        logger.info(f"📝 Model selection logged: {model_name}")
        logger.info(f"📊 Reason: {selection_reason}")
    
    def log_training_artifacts(self, model_name: str, model, history, 
                             X_train, y_train, X_val, y_val, config: dict):
        """Log training artifacts for reproducibility"""
        artifacts_dir = self.log_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save model configuration
        config_file = artifacts_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save training history
        history_file = artifacts_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history.history, f, indent=2)
        
        # Save model weights
        model_file = artifacts_dir / "model_weights.h5"
        model.save_weights(str(model_file))
        
        # Save data shapes for reproducibility
        data_info = {
            'X_train_shape': X_train.shape,
            'y_train_shape': y_train.shape,
            'X_val_shape': X_val.shape,
            'y_val_shape': y_val.shape,
            'n_features': X_train.shape[-1] if len(X_train.shape) > 1 else 1,
            'lookback_window': X_train.shape[1] if len(X_train.shape) > 2 else 1
        }
        
        data_file = artifacts_dir / "data_info.json"
        with open(data_file, 'w') as f:
            json.dump(data_info, f, indent=2)
        
        logger.info(f"💾 Training artifacts saved to {artifacts_dir}")
    
    def get_selection_summary(self) -> pd.DataFrame:
        """Get summary of all model selections"""
        if not self.selection_log:
            return pd.DataFrame()
        
        summary_data = []
        for entry in self.selection_log:
            row = {
                'timestamp': entry['timestamp'],
                'model_name': entry['model_name'],
                'rmse': entry['metrics'].get('RMSE', 0),
                'mae': entry['metrics'].get('MAE', 0),
                'mape': entry['metrics'].get('MAPE', 0),
                'r2': entry['metrics'].get('R2', 0),
                'selection_reason': entry['selection_reason']
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


class EnhancedModelSelector:
    """Enhanced model selector with comprehensive evaluation and logging"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = ModelSelectionLogger()
        self.baseline_results = {}
        self.model_results = {}
        self.selection_history = []
    
    def evaluate_with_baselines(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluate models against baseline models"""
        from baseline_models import run_baseline_comparison
        
        logger.info("🔄 Running baseline model comparison...")
        
        # Run baseline comparison
        baseline_results, baseline_df = run_baseline_comparison(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        self.baseline_results = baseline_results
        
        # Get best baseline
        best_baseline = baseline_df.iloc[0] if not baseline_df.empty else None
        if best_baseline is not None:
            logger.info(f"🏆 Best Baseline: {best_baseline['Model']} (RMSE: {best_baseline['Val_RMSE']:.4f})")
        
        return baseline_results, baseline_df
    
    def select_best_model(self, lstm_metrics: dict, transformer_metrics: dict, 
                         lstm_model=None, transformer_model=None, 
                         X_train=None, y_train=None, X_val=None, y_val=None):
        """
        Select best model using enhanced rules with logging
        
        Args:
            lstm_metrics: LSTM model metrics
            transformer_metrics: Transformer model metrics
            lstm_model: Trained LSTM model (for artifact logging)
            transformer_model: Trained Transformer model (for artifact logging)
            X_train, y_train, X_val, y_val: Training data (for artifact logging)
            
        Returns:
            tuple: (best_model_name, selection_reason, metrics_comparison)
        """
        logger.info("🎯 Starting enhanced model selection...")
        
        # Calculate RMSE difference percentage
        lstm_rmse = lstm_metrics['RMSE']
        trans_rmse = transformer_metrics['RMSE']
        rmse_diff_pct = abs(lstm_rmse - trans_rmse) / min(lstm_rmse, trans_rmse) * 100
        
        logger.info(f"📊 RMSE Comparison: LSTM={lstm_rmse:.4f}, Transformer={trans_rmse:.4f}")
        logger.info(f"📈 RMSE Difference: {rmse_diff_pct:.2f}%")
        
        # Enhanced selection rules
        if rmse_diff_pct >= 2.0:
            # Significant difference - choose better RMSE
            if lstm_rmse < trans_rmse:
                selected_model = 'lstm'
                reason = f"LSTM significantly better RMSE ({lstm_rmse:.4f} vs {trans_rmse:.4f}, diff: {rmse_diff_pct:.2f}%)"
            else:
                selected_model = 'transformer'
                reason = f"Transformer significantly better RMSE ({trans_rmse:.4f} vs {lstm_rmse:.4f}, diff: {rmse_diff_pct:.2f}%)"
        else:
            # RMSE difference < 2% - use MAE as tiebreaker
            lstm_mae = lstm_metrics['MAE']
            trans_mae = transformer_metrics['MAE']
            mae_diff_pct = abs(lstm_mae - trans_mae) / min(lstm_mae, trans_mae) * 100
            
            logger.info(f"📊 MAE Comparison: LSTM={lstm_mae:.4f}, Transformer={trans_mae:.4f}")
            logger.info(f"📈 MAE Difference: {mae_diff_pct:.2f}%")
            
            if mae_diff_pct >= 1.0:
                # MAE difference >= 1% - choose better MAE
                if lstm_mae < trans_mae:
                    selected_model = 'lstm'
                    reason = f"RMSE diff < 2%, LSTM better MAE ({lstm_mae:.4f} vs {trans_mae:.4f}, diff: {mae_diff_pct:.2f}%)"
                else:
                    selected_model = 'transformer'
                    reason = f"RMSE diff < 2%, Transformer better MAE ({trans_mae:.4f} vs {lstm_mae:.4f}, diff: {mae_diff_pct:.2f}%)"
            else:
                # MAE also close - use MAPE as final tiebreaker
                lstm_mape = lstm_metrics['MAPE']
                trans_mape = transformer_metrics['MAPE']
                
                logger.info(f"📊 MAPE Comparison: LSTM={lstm_mape:.2f}%, Transformer={trans_mape:.2f}%")
                
                if lstm_mape < trans_mape:
                    selected_model = 'lstm'
                    reason = f"RMSE & MAE diff < 2%, LSTM better MAPE ({lstm_mape:.2f}% vs {trans_mape:.2f}%)"
                else:
                    selected_model = 'transformer'
                    reason = f"RMSE & MAE diff < 2%, Transformer better MAPE ({trans_mape:.2f}% vs {lstm_mape:.2f}%)"
        
        # Get selected model metrics
        selected_metrics = lstm_metrics if selected_model == 'lstm' else transformer_metrics
        
        # Log selection decision
        self.logger.log_model_selection(
            model_name=selected_model,
            metrics=selected_metrics,
            selection_reason=reason,
            hyperparams=self.config.get(f'{selected_model}_config', {})
        )
        
        # Log training artifacts if models provided
        if selected_model == 'lstm' and lstm_model is not None:
            self.logger.log_training_artifacts(
                'lstm', lstm_model, lstm_model.history, 
                X_train, y_train, X_val, y_val, 
                self.config.get('lstm_config', {})
            )
        elif selected_model == 'transformer' and transformer_model is not None:
            self.logger.log_training_artifacts(
                'transformer', transformer_model, transformer_model.history,
                X_train, y_train, X_val, y_val,
                self.config.get('transformer_config', {})
            )
        
        # Create comparison summary
        comparison = {
            'lstm_metrics': lstm_metrics,
            'transformer_metrics': transformer_metrics,
            'rmse_difference_pct': rmse_diff_pct,
            'selected_model': selected_model,
            'selection_reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        self.selection_history.append(comparison)
        
        logger.info(f"✅ Model selection complete: {selected_model}")
        logger.info(f"📝 Reason: {reason}")
        
        return selected_model, reason, comparison
    
    def compare_with_baselines(self, model_metrics: dict, model_name: str = "Selected Model"):
        """Compare selected model with baseline models"""
        if not self.baseline_results:
            logger.warning("No baseline results available for comparison")
            return None
        
        logger.info(f"🔄 Comparing {model_name} with baseline models...")
        
        # Get best baseline metrics
        best_baseline_rmse = float('inf')
        best_baseline_name = None
        
        for name, result in self.baseline_results.items():
            if 'error' in result:
                continue
            
            val_rmse = result['val_metrics']['rmse']
            if val_rmse < best_baseline_rmse:
                best_baseline_rmse = val_rmse
                best_baseline_name = name
        
        if best_baseline_name is None:
            logger.warning("No valid baseline models found")
            return None
        
        # Compare with best baseline
        model_rmse = model_metrics['RMSE']
        improvement_pct = (best_baseline_rmse - model_rmse) / best_baseline_rmse * 100
        
        comparison_result = {
            'model_rmse': model_rmse,
            'best_baseline_name': best_baseline_name,
            'best_baseline_rmse': best_baseline_rmse,
            'improvement_pct': improvement_pct,
            'is_better': improvement_pct > 0
        }
        
        if improvement_pct > 0:
            logger.info(f"🎉 {model_name} outperforms best baseline by {improvement_pct:.2f}%")
        else:
            logger.info(f"⚠️  {model_name} underperforms best baseline by {abs(improvement_pct):.2f}%")
        
        return comparison_result
    
    def generate_selection_report(self) -> str:
        """Generate a comprehensive model selection report"""
        report_lines = [
            "# Model Selection Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Selection History",
            ""
        ]
        
        for i, selection in enumerate(self.selection_history, 1):
            report_lines.extend([
                f"### Selection {i}",
                f"**Selected Model:** {selection['selected_model']}",
                f"**Reason:** {selection['selection_reason']}",
                f"**RMSE Difference:** {selection['rmse_difference_pct']:.2f}%",
                "",
                "**LSTM Metrics:**",
                f"- RMSE: {selection['lstm_metrics']['RMSE']:.4f}",
                f"- MAE: {selection['lstm_metrics']['MAE']:.4f}",
                f"- MAPE: {selection['lstm_metrics']['MAPE']:.2f}%",
                f"- R²: {selection['lstm_metrics']['R2']:.4f}",
                "",
                "**Transformer Metrics:**",
                f"- RMSE: {selection['transformer_metrics']['RMSE']:.4f}",
                f"- MAE: {selection['transformer_metrics']['MAE']:.4f}",
                f"- MAPE: {selection['transformer_metrics']['MAPE']:.2f}%",
                f"- R²: {selection['transformer_metrics']['R2']:.4f}",
                "",
                "---",
                ""
            ])
        
        # Add baseline comparison if available
        if self.baseline_results:
            report_lines.extend([
                "## Baseline Comparison",
                ""
            ])
            
            for name, result in self.baseline_results.items():
                if 'error' in result:
                    continue
                
                val_metrics = result['val_metrics']
                report_lines.extend([
                    f"### {name}",
                    f"- RMSE: {val_metrics['rmse']:.4f}",
                    f"- MAE: {val_metrics['mae']:.4f}",
                    f"- MAPE: {val_metrics['mape']:.2f}%",
                    f"- R²: {val_metrics['r2']:.4f}",
                    ""
                ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.logger.log_dir / f"selection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 Selection report saved to {report_file}")
        
        return report_content
