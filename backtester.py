"""
Backtesting Engine for Trading Strategies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import logging
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generate trading signals based on predictions"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.trade_ledger = []
        self.entry_exit_rules = {
            'A': {
                'name': 'Simple Long Strategy',
                'entry_rule': 'predicted_return >= 0',
                'exit_rule': 'predicted_return < 0 or end_of_period',
                'description': 'Buy when AI predicts positive return, sell when negative'
            },
            'B': {
                'name': 'Target Return Strategy', 
                'entry_rule': 'predicted_return >= target_return',
                'exit_rule': 'predicted_return < target_return or end_of_period',
                'description': 'Buy when AI predicts return >= target, sell when below target'
            },
            'C': {
                'name': 'Probability-Based Strategy',
                'entry_rule': 'hit_probability >= p_min',
                'exit_rule': 'hit_probability < p_min or end_of_period', 
                'description': 'Buy when hit probability >= threshold, sell when below threshold'
            }
        }
    
    def generate_signals(self, predictions, prices, scheme='A', target_return=None, p_min=None):
        """
        Generate trading signals based on different schemes
        
        Args:
            predictions: DataFrame with predictions and probabilities
            prices: Price series
            scheme: Signal scheme ('A', 'B', 'C')
            target_return: Target return for scheme B
            p_min: Minimum probability for scheme C
            
        Returns:
            pd.DataFrame: Signals dataframe
        """
        signals = pd.DataFrame(index=predictions.index)
        signals['price'] = prices
        signals['predicted_return'] = predictions.get('predicted_return', 0)
        signals['hit_probability'] = predictions.get('hit_probability', 0)
        signals['signal'] = 0
        signals['position'] = 0
        
        if scheme == 'A':
            # Long when predicted return >= 0
            signals['signal'] = (signals['predicted_return'] >= 0).astype(int)
            
        elif scheme == 'B':
            # Long when predicted return >= target_return
            if target_return is None:
                target_return = 0
            signals['signal'] = (signals['predicted_return'] >= target_return).astype(int)
            
        elif scheme == 'C':
            # Long when hit probability >= p_min
            if p_min is None:
                p_min = 0.6
            signals['signal'] = (signals['hit_probability'] >= p_min).astype(int)
        
        # Calculate positions (long-only strategy)
        signals['position'] = signals['signal']
        
        return signals

class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.fee_bp = self.config.get('fee_bp', 15)  # 0.15%
        self.slippage_bp = self.config.get('slippage_bp', 10)  # 0.10%
        self.holding_rule = self.config.get('holding_rule', 'hold_to_horizon')
        
    def run_backtest(self, prices, signals, horizon_days, start_date=None, end_date=None):
        """
        Run backtest simulation
        
        Args:
            prices: Price series
            signals: Trading signals
            horizon_days: Holding period
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            tuple: (trades_df, equity_curve, metrics)
        """
        # Filter data by date range
        if start_date:
            mask = signals.index >= start_date
            signals = signals[mask]
            prices = prices[mask]
        
        if end_date:
            mask = signals.index <= end_date
            signals = signals[mask]
            prices = prices[mask]
        
        # Initialize tracking variables
        trades = []
        equity_curve = []
        current_position = 0
        entry_price = 0
        entry_date = None
        portfolio_value = 10000  # Starting capital
        cash = portfolio_value
        
        # Log backtest configuration
        logger.info(f"🚀 Starting backtest: scheme={scheme}, horizon={horizon_days}")
        logger.info(f"📊 Entry/Exit Rules: {self.entry_exit_rules.get(scheme, {})}")
        logger.info(f"💰 Starting capital: ${portfolio_value:,.2f}")
        logger.info(f"💸 Transaction costs: fee={self.fee_bp}bp, slippage={self.slippage_bp}bp")
        
        for i, (date, row) in enumerate(signals.iterrows()):
            current_price = row['price']
            signal = row['signal']
            
            # Check for exit conditions
            if current_position > 0:
                # Check if holding period is complete
                if self.holding_rule == 'hold_to_horizon':
                    if entry_date and (date - entry_date).days >= horizon_days:
                        # Exit position
                        exit_price = current_price
                        trade = self._create_trade(
                            entry_date, date, entry_price, exit_price, 
                            current_position, self.fee_bp, self.slippage_bp
                        )
                        trades.append(trade)
                        
                        # Update portfolio
                        portfolio_value = self._update_portfolio(
                            portfolio_value, trade, current_position
                        )
                        cash = portfolio_value
                        current_position = 0
                        entry_date = None
            
            # Check for new entry signals
            if signal == 1 and current_position == 0:
                # Enter long position
                current_position = 1
                entry_price = current_price
                entry_date = date
                cash -= current_price  # Assume we buy 1 share for simplicity
        
            # Record portfolio value
            if current_position > 0:
                current_value = cash + current_price
            else:
                current_value = cash
                
            equity_curve.append({
                'date': date,
                'portfolio_value': current_value,
                'position': current_position,
                'price': current_price
            })
        
        # Close any remaining position
        if current_position > 0:
            final_price = prices.iloc[-1]
            final_date = prices.index[-1]
            trade = self._create_trade(
                entry_date, final_date, entry_price, final_price,
                current_position, self.fee_bp, self.slippage_bp
            )
            trades.append(trade)
            portfolio_value = self._update_portfolio(
                portfolio_value, trade, current_position
            )
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades_df, equity_df)
        
        return trades_df, equity_df, metrics
    
    def _create_trade(self, entry_date, exit_date, entry_price, exit_price, 
                     position, fee_bp, slippage_bp, signal_scheme=None, 
                     predicted_return=None, hit_probability=None):
        """Create a detailed trade record with entry/exit reasoning"""
        # Calculate costs
        entry_cost = entry_price * (1 + fee_bp/10000 + slippage_bp/10000)
        exit_cost = exit_price * (1 - fee_bp/10000 - slippage_bp/10000)
        
        # Calculate returns
        gross_return = (exit_cost - entry_cost) / entry_cost
        net_return = gross_return  # Costs already included
        
        # Create detailed trade record
        trade_record = {
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position': position,
            'gross_return': gross_return,
            'net_return': net_return,
            'return_pct': net_return * 100,
            'holding_days': (exit_date - entry_date).days,
            'entry_cost': entry_cost,
            'exit_cost': exit_cost,
            'fee_bp': fee_bp,
            'slippage_bp': slippage_bp,
            'signal_scheme': signal_scheme,
            'predicted_return': predicted_return,
            'hit_probability': hit_probability,
            'entry_reason': self.entry_exit_rules.get(signal_scheme, {}).get('entry_rule', 'N/A'),
            'exit_reason': self.entry_exit_rules.get(signal_scheme, {}).get('exit_rule', 'N/A')
        }
        
        # Add to trade ledger
        self.trade_ledger.append(trade_record)
        logger.info(f"📝 Trade logged: {entry_date} -> {exit_date}, Return: {net_return*100:.2f}%")
        
        return trade_record
    
    def _update_portfolio(self, portfolio_value, trade, position):
        """Update portfolio value after trade"""
        return portfolio_value * (1 + trade['net_return'])
    
    def _calculate_metrics(self, trades_df, equity_df):
        """Calculate backtest metrics"""
        if trades_df.empty:
            return {
                'CAGR': 0,
                'Sharpe': 0,
                'Max_Drawdown': 0,
                'Volatility': 0,
                'Win_Rate': 0,
                'Avg_Gain': 0,
                'Avg_Loss': 0,
                'Profit_Factor': 0,
                'Hit_Ratio': 0,
                'N_Trades': 0,
                'Turnover': 0
            }
        
        # Basic metrics
        total_return = (equity_df['portfolio_value'].iloc[-1] / equity_df['portfolio_value'].iloc[0]) - 1
        years = len(equity_df) / 252
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Returns
        returns = equity_df['portfolio_value'].pct_change().dropna()
        
        # Sharpe ratio
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Maximum drawdown
        peak = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Trade metrics
        winning_trades = trades_df[trades_df['net_return'] > 0]
        losing_trades = trades_df[trades_df['net_return'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_gain = winning_trades['net_return'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_return'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['net_return'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['net_return'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Hit ratio (directional accuracy)
        hit_ratio = win_rate
        
        # Turnover (simplified)
        turnover = len(trades_df) / years if years > 0 else 0
        
        return {
            'CAGR': cagr * 100,
            'Sharpe': sharpe,
            'Max_Drawdown': max_drawdown * 100,
            'Volatility': volatility * 100,
            'Win_Rate': win_rate * 100,
            'Avg_Gain': avg_gain * 100,
            'Avg_Loss': avg_loss * 100,
            'Profit_Factor': profit_factor,
            'Hit_Ratio': hit_ratio * 100,
            'N_Trades': len(trades_df),
            'Turnover': turnover
        }

class WalkForwardBacktester:
    """Walk-forward backtesting"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.train_months = self.config.get('walk_forward_train_months', 36)
        self.test_months = self.config.get('walk_forward_test_months', 6)
        self.step_months = self.config.get('walk_forward_step_months', 6)
    
    def run_walk_forward(self, prices, signals, horizon_days):
        """
        Run walk-forward backtesting
        
        Args:
            prices: Price series
            signals: Trading signals
            horizon_days: Holding period
            
        Returns:
            dict: Walk-forward results
        """
        backtester = Backtester(self.config)
        results = []
        
        # Calculate date ranges
        start_date = prices.index[0]
        end_date = prices.index[-1]
        
        current_date = start_date
        while current_date < end_date:
            # Training period
            train_start = current_date
            train_end = current_date + pd.DateOffset(months=self.train_months)
            
            # Test period
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            if test_end > end_date:
                break
            
            # Run backtest for this period
            trades_df, equity_df, metrics = backtester.run_backtest(
                prices, signals, horizon_days, test_start, test_end
            )
            
            results.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'trades': trades_df,
                'equity': equity_df,
                'metrics': metrics
            })
            
            # Move to next period
            current_date += pd.DateOffset(months=self.step_months)
        
        return results

class BacktestAnalyzer:
    """Analyze backtest results"""
    
    @staticmethod
    def create_confusion_matrix(trades_df):
        """Create confusion matrix for directional accuracy"""
        if trades_df.empty:
            return {
                'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
                'Precision': 0, 'Recall': 0, 'F1': 0
            }
        
        # For simplicity, assume all trades are long positions
        # In a more sophisticated implementation, we'd track actual vs predicted direction
        winning_trades = len(trades_df[trades_df['net_return'] > 0])
        total_trades = len(trades_df)
        
        # Simplified confusion matrix
        tp = winning_trades  # True positives (correct long predictions)
        fp = total_trades - winning_trades  # False positives (incorrect long predictions)
        tn = 0  # True negatives (no short positions in this strategy)
        fn = 0  # False negatives (no short positions in this strategy)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
    @staticmethod
    def calculate_rolling_metrics(equity_df, window=252):
        """Calculate rolling metrics"""
        returns = equity_df['portfolio_value'].pct_change().dropna()
        
        rolling_sharpe = returns.rolling(window=window).mean() / returns.rolling(window=window).std() * np.sqrt(252)
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        return {
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_vol
        }
