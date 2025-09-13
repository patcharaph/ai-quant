"""
Artifact Logger for Training Results and Model Selection
"""

import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

class ArtifactLogger:
    """Logs training results, model selection reasoning, and artifacts to JSON files"""
    
    def __init__(self, base_dir="artifacts"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_training_results(self, model_name, results, metadata=None):
        """Log training results for a specific model"""
        artifact = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "results": results,
            "metadata": metadata or {}
        }
        
        filename = f"{self.session_id}_{model_name}_training.json"
        filepath = self.base_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 Training results logged: {filepath}")
        return filepath
    
    def log_model_comparison(self, comparison_results, selected_model, reasoning):
        """Log model comparison and selection reasoning"""
        artifact = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "comparison_results": comparison_results,
            "selected_model": selected_model,
            "selection_reasoning": reasoning,
            "selection_criteria": {
                "primary": "RMSE (lower is better)",
                "secondary": "MAE (lower is better)", 
                "tertiary": "MAPE (lower is better)",
                "tie_breaker": "R² (higher is better)"
            }
        }
        
        filename = f"{self.session_id}_model_comparison.json"
        filepath = self.base_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"🏆 Model comparison logged: {filepath}")
        return filepath
    
    def log_prediction_results(self, predictions, risk_metrics, advisory, metadata=None):
        """Log prediction results and risk analysis"""
        artifact = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "risk_metrics": risk_metrics,
            "advisory": advisory,
            "metadata": metadata or {}
        }
        
        filename = f"{self.session_id}_predictions.json"
        filepath = self.base_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"🔮 Prediction results logged: {filepath}")
        return filepath
    
    def log_backtest_results(self, backtest_results, trade_ledger, metadata=None):
        """Log backtest results and trade ledger"""
        artifact = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "backtest_results": backtest_results,
            "trade_ledger": trade_ledger,
            "metadata": metadata or {}
        }
        
        filename = f"{self.session_id}_backtest.json"
        filepath = self.base_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📈 Backtest results logged: {filepath}")
        return filepath
    
    def log_session_summary(self, session_data):
        """Log complete session summary"""
        artifact = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "session_summary": session_data
        }
        
        filename = f"{self.session_id}_session_summary.json"
        filepath = self.base_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📋 Session summary logged: {filepath}")
        return filepath
    
    def get_session_files(self):
        """Get all files for current session"""
        pattern = f"{self.session_id}_*.json"
        return list(self.base_dir.glob(pattern))
    
    def cleanup_old_artifacts(self, days_to_keep=30):
        """Clean up old artifact files"""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        for file_path in self.base_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_date:
                file_path.unlink()
                print(f"🗑️  Cleaned up old artifact: {file_path.name}")
