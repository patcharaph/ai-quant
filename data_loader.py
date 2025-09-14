"""
Data Loader for fetching and preprocessing OHLCV data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import pytz
import re
import os
import time
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
warnings.filterwarnings('ignore')

class DataLoader:
    """Handles data fetching and basic preprocessing"""
    
    def __init__(self, cache_dir='data_cache'):
        self.data_cache = {}
        self.bangkok_tz = pytz.timezone('Asia/Bangkok')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Thai stock symbol mapping
        self.thai_symbols = {
            # Major Thai stocks
            'PTT': 'PTT.BK', 'SCB': 'SCB.BK', 'KBANK': 'KBANK.BK', 'KBANK.BK': 'KBANK.BK',
            'CPALL': 'CPALL.BK', 'ADVANC': 'ADVANC.BK', 'AOT': 'AOT.BK', 'BDMS': 'BDMS.BK',
            'CPF': 'CPF.BK', 'CPN': 'CPN.BK', 'DELTA': 'DELTA.BK', 'EGCO': 'EGCO.BK',
            'GLOBAL': 'GLOBAL.BK', 'HANA': 'HANA.BK', 'INTUCH': 'INTUCH.BK', 'JAS': 'JAS.BK',
            'KBANK': 'KBANK.BK', 'KTB': 'KTB.BK', 'LH': 'LH.BK', 'MINT': 'MINT.BK',
            'PTTEP': 'PTTEP.BK', 'PTTGC': 'PTTGC.BK', 'RATCH': 'RATCH.BK', 'SCC': 'SCC.BK',
            'TCAP': 'TCAP.BK', 'TMB': 'TMB.BK', 'TOP': 'TOP.BK', 'TRUE': 'TRUE.BK',
            'TTB': 'TTB.BK', 'TU': 'TU.BK', 'WHA': 'WHA.BK',
            # Indexes
            'SET': '^SET.BK', 'SET50': '^SET50.BK', 'SET100': '^SET100.BK',
            # US stocks (if traded in Thai market)
            'AAPL': 'AAPL.BK', 'GOOGL': 'GOOGL.BK', 'MSFT': 'MSFT.BK', 
            'TSLA': 'TSLA.BK', 'AMZN': 'AMZN.BK', 'META': 'META.BK'
        }
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.timeout = 30  # seconds
        
        # Rate limiting configuration
        self.rate_limit_delay = 2  # seconds between requests
        self.last_request_time = 0
    
    def validate_symbol(self, symbol):
        """
        Validate and normalize stock symbol for Thai market
        
        Args:
            symbol (str): Input symbol (e.g., 'PTT', 'PTT.BK', 'SET')
            
        Returns:
            str: Normalized symbol with .BK suffix
            
        Raises:
            ValueError: If symbol cannot be mapped or validated
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Clean the symbol
        symbol = symbol.strip().upper()
        
        # Check if already has .BK suffix
        if symbol.endswith('.BK'):
            return symbol
        
        # Check if it's an index
        if symbol.startswith('^'):
            return symbol
        
        # Map Thai symbols
        if symbol in self.thai_symbols:
            mapped_symbol = self.thai_symbols[symbol]
            print(f"✓ Mapped '{symbol}' to '{mapped_symbol}'")
            return mapped_symbol
        
        # For unknown symbols, try adding .BK suffix
        normalized_symbol = f"{symbol}.BK"
        print(f"⚠️  Unknown symbol '{symbol}', trying '{normalized_symbol}'")
        return normalized_symbol
    
    def _get_cache_path(self, symbol, start_date, end_date, interval):
        """Get cache file path for data"""
        cache_filename = f"{symbol}_{start_date}_{end_date}_{interval}.parquet"
        return self.cache_dir / cache_filename
    
    def _load_from_cache(self, cache_path):
        """Load data from cache if available and not expired"""
        if not cache_path.exists():
            return None
        
        try:
            # Check if cache is less than 1 hour old
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > 3600:  # 1 hour
                return None
            
            data = pd.read_parquet(cache_path)
            # Ensure timezone is preserved
            if hasattr(data.index, 'tz') and data.index.tz is None:
                data.index = data.index.tz_localize('Asia/Bangkok')
            elif hasattr(data.index, 'tz') and data.index.tz != self.bangkok_tz:
                data.index = data.index.tz_convert(self.bangkok_tz)
            
            print(f"✓ Loaded data from cache: {cache_path.name}")
            return data
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}")
            return None
    
    def _apply_rate_limiting(self):
        """Apply rate limiting to prevent API throttling"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            print(f"⏳ Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _test_symbol_availability(self, symbol: str) -> bool:
        """Test if symbol is available and accessible"""
        try:
            self._apply_rate_limiting()
            ticker = yf.Ticker(symbol)
            
            # Try to get basic info first
            info = ticker.info
            if not info or 'symbol' not in info:
                return False
            
            # Try to get recent data
            data = ticker.history(period='1d')
            return not data.empty
            
        except Exception as e:
            print(f"⚠️  Symbol {symbol} test failed: {str(e)[:50]}...")
            return False
    
    def _save_to_cache(self, data, cache_path):
        """Save data to cache"""
        try:
            data.to_parquet(cache_path)
            print(f"✓ Cached data: {cache_path.name}")
        except Exception as e:
            print(f"⚠️  Error saving to cache: {e}")

    def fetch_ohlcv(self, symbol, start_date=None, end_date=None, interval="1d"):
        """
        Fetch OHLCV data for a given symbol with enhanced retry logic and caching
        
        Args:
            symbol (str): Stock symbol (e.g., 'PTT', 'AAPL') - will be normalized to .BK
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval (1d, 1h, etc.)
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index in Asia/Bangkok timezone
            
        Raises:
            ValueError: If symbol cannot be fetched or is invalid
        """
        try:
            # Validate and normalize symbol
            original_symbol = symbol
            symbol = self.validate_symbol(symbol)
            
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now(self.bangkok_tz).strftime('%Y-%m-%d')
            
            if start_date is None:
                start_date = (datetime.now(self.bangkok_tz) - timedelta(days=20*365)).strftime('%Y-%m-%d')
            
            # Check cache first
            cache_path = self._get_cache_path(symbol, start_date, end_date, interval)
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
            
            # Test symbol availability first
            if not self._test_symbol_availability(symbol):
                raise ValueError(f"สัญลักษณ์ {original_symbol} ไม่สามารถเข้าถึงได้หรือไม่มีข้อมูล")
            
            # Fetch data with enhanced retry logic
            data = None
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    print(f"🔄 Fetching data for {symbol} (attempt {attempt + 1}/{self.max_retries})")
                    
                    # Apply rate limiting
                    self._apply_rate_limiting()
                    
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date, interval=interval)
                    
                    if not data.empty:
                        print(f"✅ Successfully fetched {len(data)} rows for {symbol}")
                        break
                    else:
                        print(f"⚠️  No data returned for {symbol} in date range {start_date} to {end_date}")
                        
                        # Try with different date ranges if no data
                        if attempt == 0:
                            # Try with shorter period first
                            start_date_alt = (datetime.now(self.bangkok_tz) - timedelta(days=365)).strftime('%Y-%m-%d')
                            print(f"🔄 Trying shorter period: {start_date_alt} to {end_date}")
                            data = ticker.history(start=start_date_alt, end=end_date, interval=interval)
                            
                            if not data.empty:
                                start_date = start_date_alt
                                break
                        elif attempt == 1:
                            # Try with even shorter period
                            start_date_alt = (datetime.now(self.bangkok_tz) - timedelta(days=90)).strftime('%Y-%m-%d')
                            print(f"🔄 Trying 3-month period: {start_date_alt} to {end_date}")
                            data = ticker.history(start=start_date_alt, end=end_date, interval=interval)
                            
                            if not data.empty:
                                start_date = start_date_alt
                                break
                        
                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__
                    print(f"⚠️  Attempt {attempt + 1} failed ({error_type}): {str(e)[:100]}...")
                    
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
            
            if data is None or data.empty:
                error_msg = f"ไม่สามารถดึงข้อมูลสำหรับ {original_symbol} ได้"
                if last_error:
                    error_msg += f" (Error: {str(last_error)[:100]}...)"
                else:
                    error_msg += " (ไม่มีข้อมูลในช่วงเวลาที่ระบุ)"
                
                print(f"❌ {error_msg}")
                print(f"💡 ตรวจสอบสัญลักษณ์หุ้นและช่วงวันที่ หรือลองใช้สัญลักษณ์อื่น")
                
                # Provide helpful suggestions
                if '.BK' in symbol:
                    print(f"💡 สำหรับหุ้นไทย ลองใช้: {symbol}")
                else:
                    print(f"💡 ลองใช้สัญลักษณ์อื่น เช่น: PTT.BK, SCB.BK, KBANK.BK")
                
                raise ValueError(error_msg)
            
            # Clean and process data
            data = self._process_fetched_data(data, symbol)
            
            # Save to cache
            self._save_to_cache(data, cache_path)
            
            return data
            
        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            error_msg = f"เกิดข้อผิดพลาดในการดึงข้อมูลสำหรับ {original_symbol}: {str(e)[:100]}..."
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
    
    def _process_fetched_data(self, data, symbol):
        """Process and clean fetched data with enhanced error handling"""
        try:
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"⚠️  Missing columns: {missing_cols}")
                # Try to map common variations
                column_mapping = {
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
                    'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume'
                }
                for old_col, new_col in column_mapping.items():
                    if old_col in data.columns and new_col not in data.columns:
                        data[new_col] = data[old_col]
            
            # Convert to Asia/Bangkok timezone
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_convert(self.bangkok_tz)
            else:
                try:
                    data.index = data.index.tz_localize('UTC').tz_convert(self.bangkok_tz)
                except:
                    # If timezone conversion fails, assume it's already in Bangkok time
                    data.index = data.index.tz_localize(self.bangkok_tz)
            
            # Add timezone info to metadata
            data.attrs['timezone'] = 'Asia/Bangkok'
            data.attrs['symbol'] = symbol
            data.attrs['fetched_at'] = datetime.now(self.bangkok_tz).isoformat()
            data.attrs['data_source'] = 'yfinance'
            
            # Validate data quality
            validation_results = self.validate_data(data)
            if not validation_results['is_valid']:
                print("⚠️  Data quality warnings:")
                for warning in validation_results['warnings']:
                    print(f"   - {warning}")
            
            return data
            
        except Exception as e:
            print(f"⚠️  Error processing data: {e}")
            # Return data as-is if processing fails
            return data
    
    def validate_data(self, data, min_years=3):
        """
        Validate data quality and quantity
        
        Args:
            data (pd.DataFrame): OHLCV data
            min_years (int): Minimum years of data required
            
        Returns:
            dict: Validation results with warnings and recommendations
        """
        results = {
            'is_valid': True,
            'warnings': [],
            'recommendations': [],
            'data_quality': {}
        }
        
        # Check data length
        years_available = len(data) / 252  # Approximate trading days per year
        results['data_quality']['years_available'] = years_available
        
        if years_available < min_years:
            results['warnings'].append(f"Data available for only {years_available:.1f} years (minimum: {min_years})")
            results['is_valid'] = False
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        results['data_quality']['missing_pct'] = missing_pct
        
        if missing_pct > 5:
            results['warnings'].append(f"High missing data: {missing_pct:.1f}%")
        
        # Check for zero volume days
        zero_volume_pct = (data['volume'] == 0).sum() / len(data) * 100
        results['data_quality']['zero_volume_pct'] = zero_volume_pct
        
        if zero_volume_pct > 10:
            results['warnings'].append(f"High zero volume days: {zero_volume_pct:.1f}%")
            results['recommendations'].append("Consider filtering out zero volume days")
        
        # Check for price anomalies
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                # Check for negative prices
                negative_prices = (data[col] <= 0).sum()
                if negative_prices > 0:
                    results['warnings'].append(f"Found {negative_prices} non-positive {col} prices")
                
                # Check for extreme price changes
                returns = data[col].pct_change().dropna()
                extreme_changes = (abs(returns) > 0.5).sum()  # >50% daily change
                if extreme_changes > 0:
                    results['warnings'].append(f"Found {extreme_changes} extreme price changes in {col}")
        
        return results
    
    def clean_data(self, data):
        """
        Clean and preprocess the data with enhanced error handling
        
        Args:
            data (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            cleaned_data = data.copy()
            
            # Remove rows with missing critical data
            critical_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in critical_cols if col in cleaned_data.columns]
            
            if not available_cols:
                print("⚠️  No critical columns found for cleaning")
                return cleaned_data
            
            cleaned_data = cleaned_data.dropna(subset=available_cols)
            
            # Remove rows with zero volume (if not too many)
            if 'volume' in cleaned_data.columns:
                zero_volume_pct = (cleaned_data['volume'] == 0).sum() / len(cleaned_data) * 100
                if zero_volume_pct < 20:  # Only remove if not too many
                    cleaned_data = cleaned_data[cleaned_data['volume'] > 0]
            
            # Ensure price relationships are valid
            if all(col in cleaned_data.columns for col in ['high', 'low']):
                cleaned_data = cleaned_data[cleaned_data['high'] >= cleaned_data['low']]
            
            if all(col in cleaned_data.columns for col in ['high', 'open', 'close', 'low']):
                cleaned_data = cleaned_data[
                    (cleaned_data['high'] >= cleaned_data['open']) &
                    (cleaned_data['high'] >= cleaned_data['close']) &
                    (cleaned_data['low'] <= cleaned_data['open']) &
                    (cleaned_data['low'] <= cleaned_data['close'])
                ]
            
            # Forward fill any remaining missing values
            cleaned_data = cleaned_data.fillna(method='ffill')
            
            print(f"✅ Cleaned data: {len(data)} -> {len(cleaned_data)} rows")
            return cleaned_data
            
        except Exception as e:
            print(f"⚠️  Error cleaning data: {e}")
            return data
    
    def get_data_summary(self, data):
        """
        Get summary statistics of the data with enhanced error handling
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            dict: Summary statistics
        """
        try:
            summary = {
                'start_date': data.index[0].strftime('%Y-%m-%d') if len(data) > 0 else 'N/A',
                'end_date': data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else 'N/A',
                'total_days': len(data),
                'trading_days': len(data[data['volume'] > 0]) if 'volume' in data.columns else len(data),
                'years_covered': len(data) / 252,
                'price_range': {},
                'volume_stats': {}
            }
            
            # Price range statistics
            if 'close' in data.columns:
                summary['price_range'] = {
                    'min': float(data['close'].min()),
                    'max': float(data['close'].max()),
                    'current': float(data['close'].iloc[-1]) if len(data) > 0 else 0
                }
            
            # Volume statistics
            if 'volume' in data.columns:
                summary['volume_stats'] = {
                    'mean': float(data['volume'].mean()),
                    'median': float(data['volume'].median()),
                    'std': float(data['volume'].std())
                }
            
            return summary
            
        except Exception as e:
            print(f"⚠️  Error generating summary: {e}")
            return {
                'start_date': 'N/A',
                'end_date': 'N/A',
                'total_days': len(data) if data is not None else 0,
                'trading_days': 0,
                'years_covered': 0,
                'price_range': {'min': 0, 'max': 0, 'current': 0},
                'volume_stats': {'mean': 0, 'median': 0, 'std': 0}
            }
    
    def get_available_symbols(self) -> Dict[str, str]:
        """
        Get list of available Thai stock symbols
        
        Returns:
            dict: Dictionary of symbol mappings
        """
        return self.thai_symbols.copy()
    
    def test_connection(self) -> bool:
        """
        Test connection to data source
        
        Returns:
            bool: True if connection is working
        """
        try:
            # Test with a simple US stock first
            test_symbol = 'AAPL'
            print(f"🔍 Testing connection with {test_symbol}...")
            
            if self._test_symbol_availability(test_symbol):
                print("✅ Connection test successful")
                return True
            else:
                print("❌ Connection test failed")
                return False
                
        except Exception as e:
            print(f"❌ Connection test error: {e}")
            return False
