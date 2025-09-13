"""
LSTM and Transformer models for time series prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class LSTMModel:
    """LSTM model for time series prediction"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential()
        
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
