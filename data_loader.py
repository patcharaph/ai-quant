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
warnings.filterwarnings('ignore')

class DataLoader:
    """Handles data fetching and basic preprocessing"""
    
    def __init__(self):
        self.data_cache = {}
        self.bangkok_tz = pytz.timezone('Asia/Bangkok')
        self.valid_symbols = {
            # Thai stocks
            'SET.BK', 'SET50.BK', 'PTT.BK', 'SCB.BK', 'KBANK.BK', 
            'CPALL.BK', 'ADVANC.BK', 'AOT.BK', 'BDMS.BK', 'CPF.BK',
            # US stocks (with .BK suffix for Thai market)
            'AAPL.BK', 'GOOGL.BK', 'MSFT.BK', 'TSLA.BK', 'AMZN.BK', 'META.BK'
        }
    
    def validate_symbol(self, symbol):
        """Validate and normalize stock symbol"""
        # Remove any existing .BK suffix
        base_symbol = symbol.replace('.BK', '')
        
        # Add .BK suffix for Thai market
        normalized_symbol = f"{base_symbol}.BK"
        
        # Check if symbol is in valid list
        if normalized_symbol not in self.valid_symbols:
            print(f"Warning: {normalized_symbol} not in predefined valid symbols list")
            print(f"Valid symbols: {', '.join(sorted(self.valid_symbols))}")
        
        return normalized_symbol

    def fetch_ohlcv(self, symbol, start_date=None, end_date=None, interval="1d"):
        """
        Fetch OHLCV data for a given symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL') - will be normalized to .BK
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval (1d, 1h, etc.)
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index in Asia/Bangkok timezone
        """
        try:
            # Validate and normalize symbol
            symbol = self.validate_symbol(symbol)
            
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now(self.bangkok_tz).strftime('%Y-%m-%d')
            
            if start_date is None:
                start_date = (datetime.now(self.bangkok_tz) - timedelta(days=20*365)).strftime('%Y-%m-%d')
            
            # Check cache first
            cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key].copy()
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                # Try with different date range if no data
                print(f"Warning: No data found for {symbol} in date range {start_date} to {end_date}")
                # Try with last 5 years
                start_date_alt = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
                data = ticker.history(start=start_date_alt, end=end_date, interval=interval)
                
                if data.empty:
                    raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            
            # Convert to Asia/Bangkok timezone
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_convert(self.bangkok_tz)
            else:
                data.index = data.index.tz_localize('UTC').tz_convert(self.bangkok_tz)
            
            # Add timezone info to metadata
            data.attrs['timezone'] = 'Asia/Bangkok'
            data.attrs['symbol'] = symbol
            data.attrs['fetched_at'] = datetime.now(self.bangkok_tz).isoformat()
            
            # Cache the data
            self.data_cache[cache_key] = data.copy()
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
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
        Clean and preprocess the data
        
        Args:
            data (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        cleaned_data = data.copy()
        
        # Remove rows with missing critical data
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        cleaned_data = cleaned_data.dropna(subset=critical_cols)
        
        # Remove rows with zero volume (if too many)
        zero_volume_pct = (cleaned_data['volume'] == 0).sum() / len(cleaned_data) * 100
        if zero_volume_pct < 20:  # Only remove if not too many
            cleaned_data = cleaned_data[cleaned_data['volume'] > 0]
        
        # Ensure high >= low
        cleaned_data = cleaned_data[cleaned_data['high'] >= cleaned_data['low']]
        
        # Ensure high >= open, close and low <= open, close
        cleaned_data = cleaned_data[
            (cleaned_data['high'] >= cleaned_data['open']) &
            (cleaned_data['high'] >= cleaned_data['close']) &
            (cleaned_data['low'] <= cleaned_data['open']) &
            (cleaned_data['low'] <= cleaned_data['close'])
        ]
        
        # Forward fill any remaining missing values
        cleaned_data = cleaned_data.fillna(method='ffill')
        
        return cleaned_data
    
    def get_data_summary(self, data):
        """
        Get summary statistics of the data
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'total_days': len(data),
            'trading_days': len(data[data['volume'] > 0]),
            'years_covered': len(data) / 252,
            'price_range': {
                'min': data['close'].min(),
                'max': data['close'].max(),
                'current': data['close'].iloc[-1]
            },
            'volume_stats': {
                'mean': data['volume'].mean(),
                'median': data['volume'].median(),
                'std': data['volume'].std()
            }
        }
        
        return summary
