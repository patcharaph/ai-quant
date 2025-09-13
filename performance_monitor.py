"""
Performance Monitoring and Optimization Utilities

This module provides performance monitoring, profiling, and optimization tools
for the AI Quant Stock Prediction System.
"""

import time
import psutil
import gc
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.memory_usage = []
        self.cpu_usage = []
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.time() - self.start_times[name]
        self.metrics[name] = duration
        del self.start_times[name]
        
        logger.info(f"⏱️  {name}: {duration:.3f}s")
        return duration
    
    def record_memory_usage(self):
        """Record current memory usage"""
        memory_info = psutil.virtual_memory()
        self.memory_usage.append({
            'timestamp': time.time(),
            'used_gb': memory_info.used / (1024**3),
            'available_gb': memory_info.available / (1024**3),
            'percent': memory_info.percent
        })
    
    def record_cpu_usage(self):
        """Record current CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.append({
            'timestamp': time.time(),
            'cpu_percent': cpu_percent
        })
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        memory_info = psutil.virtual_memory()
        return {
            'total_gb': memory_info.total / (1024**3),
            'available_gb': memory_info.available / (1024**3),
            'used_gb': memory_info.used / (1024**3),
            'percent': memory_info.percent
        }
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU statistics"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    
    def cleanup_memory(self):
        """Force garbage collection to free memory"""
        gc.collect()
        logger.info("🧹 Memory cleanup completed")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'timings': self.metrics,
            'memory_stats': self.get_memory_stats(),
            'cpu_stats': self.get_cpu_stats(),
            'total_operations': len(self.metrics)
        }

# Global performance monitor
perf_monitor = PerformanceMonitor()

def time_function(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        perf_monitor.start_timer(func_name)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            perf_monitor.end_timer(func_name)
    return wrapper

def monitor_memory(func: Callable) -> Callable:
    """Decorator to monitor memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Record memory before
        memory_before = perf_monitor.get_memory_stats()
        
        result = func(*args, **kwargs)
        
        # Record memory after
        memory_after = perf_monitor.get_memory_stats()
        
        memory_diff = memory_after['used_gb'] - memory_before['used_gb']
        logger.info(f"💾 Memory usage change: {memory_diff:+.3f} GB")
        
        return result
    return wrapper

class DataOptimizer:
    """Optimize data processing operations"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        # Optimize categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Low cardinality
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"📊 DataFrame optimized: {reduction:.1f}% memory reduction")
        return df
    
    @staticmethod
    def chunk_processing(data, chunk_size: int = 1000, func: Callable = None):
        """Process data in chunks to manage memory"""
        if func is None:
            return data
        
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            result = func(chunk)
            results.append(result)
            
            # Cleanup
            del chunk
            gc.collect()
        
        return results

class ModelOptimizer:
    """Optimize model training and inference"""
    
    @staticmethod
    def optimize_tensorflow():
        """Optimize TensorFlow settings"""
        try:
            import tensorflow as tf
            
            # Set memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision if available
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logger.info("🚀 Mixed precision enabled")
            except Exception:
                pass
            
            # Optimize for performance
            tf.config.optimizer.set_jit(True)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            tf.config.threading.set_inter_op_parallelism_threads(0)
            
            logger.info("✅ TensorFlow optimized")
            
        except ImportError:
            logger.warning("⚠️  TensorFlow not available for optimization")
    
    @staticmethod
    def batch_predictions(model, X, batch_size: int = 32):
        """Make predictions in batches to manage memory"""
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            batch_pred = model.predict(batch, verbose=0)
            predictions.append(batch_pred)
            
            # Cleanup
            del batch, batch_pred
            gc.collect()
        
        return np.concatenate(predictions, axis=0)

def profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile a function and return performance metrics"""
    import cProfile
    import pstats
    import io
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    # Get profiling stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    return {
        'result': result,
        'profile_stats': s.getvalue(),
        'total_calls': ps.total_calls,
        'primitive_calls': ps.prim_calls
    }

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else None,
        'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        'platform': psutil.sys.platform
    }

# Initialize optimizations
def initialize_optimizations():
    """Initialize all performance optimizations"""
    logger.info("🚀 Initializing performance optimizations...")
    
    # Optimize TensorFlow
    ModelOptimizer.optimize_tensorflow()
    
    # Set NumPy threading
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Record initial system state
    perf_monitor.record_memory_usage()
    perf_monitor.record_cpu_usage()
    
    logger.info("✅ Performance optimizations initialized")

# Auto-initialize when module is imported
import os
initialize_optimizations()
