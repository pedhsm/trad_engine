"""
Trade metrics for extracting entry/exit signals from TradingStrategy models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class TradeSignal:
    """Standardised trade signal with entry/exit information."""
    timestamp: pd.Timestamp
    signal_type: str  # entry or exit
    position: str  # long, short, or flat
    price: float
    metadata: Dict[str, Any]


def extract_entry_exit_signals(strategy_result, ohlc: pd.DataFrame,
                              position_threshold: float = 0.0) -> List[TradeSignal]:
    """
    Extract entry/exit signals from TradingStrategy StrategyResult in plug-and-play manner.

    Compatible with the TradingStrategy base class pattern from validation.strategy_base.

    Args:
        strategy_result: StrategyResult object from TradingStrategy.generate_signal()
        ohlc: DataFrame with OHLC data (normalised columns: open, high, low, close)
        position_threshold: Threshold for determining position changes (default: 0.0)

    Returns:
        List of TradeSignal objects with entry/exit information
    """
    # Validate inputs follow TradingStrategy pattern
    if not hasattr(strategy_result, 'signal') or not hasattr(strategy_result, 'metadata'):
        raise ValueError("strategy_result must be a StrategyResult object with 'signal' and 'metadata' attributes")

    if not isinstance(strategy_result.signal, pd.Series):
        raise ValueError("strategy_result.signal must be a pandas Series")

    # Normalise OHLC columns (following TradingStrategy._normalise_ohlc pattern)
    ohlc = ohlc.copy()
    ohlc.columns = ohlc.columns.str.lower()
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in ohlc.columns]
    if missing_cols:
        raise ValueError(f"OHLC data must contain columns: {required_cols}. Missing: {missing_cols}")

    signals = []
    signal_series = strategy_result.signal

    # Determine position states based on signal values
    current_position = 'flat'

    for i, (timestamp, signal_value) in enumerate(signal_series.items()):
        if timestamp not in ohlc.index:
            continue

        # Determine new position based on signal
        if signal_value > position_threshold:
            new_position = 'long'
        elif signal_value < -position_threshold:
            new_position = 'short'
        else:
            new_position = 'flat'

        # Detect position changes (entry/exit logic)
        if new_position != current_position:
            # Exit previous position (if not flat)
            if current_position != 'flat':
                exit_signal = TradeSignal(
                    timestamp=timestamp,
                    signal_type='exit',
                    position=current_position,
                    price=ohlc.loc[timestamp, 'close'],
                    metadata={
                        'strategy_name': strategy_result.metadata.get('strategy_name', 'unknown'),
                        'exit_signal_value': signal_value,
                        'previous_position': current_position,
                        **strategy_result.metadata
                    }
                )
                signals.append(exit_signal)

            # Enter new position (if not flat)
            if new_position != 'flat':
                entry_signal = TradeSignal(
                    timestamp=timestamp,
                    signal_type='entry',
                    position=new_position,
                    price=ohlc.loc[timestamp, 'close'],
                    metadata={
                        'strategy_name': strategy_result.metadata.get('strategy_name', 'unknown'),
                        'entry_signal_value': signal_value,
                        'position_size': abs(signal_value),
                        **strategy_result.metadata
                    }
                )
                signals.append(entry_signal)

            current_position = new_position

    return signals


def get_model_trades(strategy_result, ohlc: pd.DataFrame,
                    position_threshold: float = 0.0) -> pd.DataFrame:
    """
    Convert TradingStrategy signals to trade dataframe with entry/exit pairs.

    Plug-and-play function compatible with TradingStrategy base class pattern.

    Args:
        strategy_result: StrategyResult object from TradingStrategy.generate_signal()
        ohlc: DataFrame with OHLC data
        position_threshold: Threshold for determining position changes

    Returns:
        DataFrame with columns: entry_time, exit_time, entry_price, exit_price,
                               position, direction, trade_size, pnl, return_pct, result,
                               holding_time, holding_time_hours, strategy_name
    """
    signals = extract_entry_exit_signals(strategy_result, ohlc, position_threshold)

    trades = []
    open_trades = {}  # Track open positions by position type

    for signal in signals:
        if signal.signal_type == 'entry':
            # Store entry signal
            open_trades[signal.position] = signal

        elif signal.signal_type == 'exit' and signal.position in open_trades:
            # Match with corresponding entry
            entry_signal = open_trades.pop(signal.position)

            # Calculate PnL
            if signal.position == 'long':
                pnl = signal.price - entry_signal.price
            else:  # short position
                pnl = entry_signal.price - signal.price

            # Calculate holding time
            holding_time = signal.timestamp - entry_signal.timestamp
            holding_time_hours = _calculate_holding_time_hours(holding_time)

            # Determine trade result (break-even trades count as LOSS)
            if pnl > 0:
                result = 'WIN'
            elif pnl < 0:
                result = 'LOSS'
            else:
                result = 'BREAKEVEN'

            # Get trade size from entry signal metadata or use absolute signal value
            trade_size = entry_signal.metadata.get('position_size', abs(entry_signal.metadata.get('entry_signal_value', 1.0)))

            # Fixed: Return percentage calculation
            # For longs: (exit - entry) / entry
            # For shorts: (entry - exit) / entry (same as pnl / entry)
            # Both cases: pnl / entry_price works correctly
            if entry_signal.price > 0:
                return_pct = pnl / entry_signal.price
            else:
                # Handle edge case of zero entry price
                return_pct = 0.0

            trade = {
                'entry_time': entry_signal.timestamp,
                'exit_time': signal.timestamp,
                'entry_price': entry_signal.price,
                'exit_price': signal.price,
                'position': signal.position,
                'direction': signal.position.upper(),  # 'LONG' or 'SHORT'
                'trade_size': trade_size,
                'pnl': pnl,
                'return_pct': return_pct,
                'result': result,
                'holding_time': holding_time,
                'holding_time_hours': holding_time_hours,
                'strategy_name': signal.metadata.get('strategy_name', 'unknown'),
                'entry_signal_value': entry_signal.metadata.get('entry_signal_value', 0),
                'exit_signal_value': signal.metadata.get('exit_signal_value', 0)
            }
            trades.append(trade)

    return pd.DataFrame(trades)


def analyse_model_performance(strategy_result, ohlc: pd.DataFrame,
                             position_threshold: float = 0.0) -> Dict[str, Any]:
    """
    Comprehensive performance analysis for TradingStrategy models.

    Plug-and-play function following TradingStrategy framework patterns.

    Args:
        strategy_result: StrategyResult object from TradingStrategy.generate_signal()
        ohlc: DataFrame with OHLC data
        position_threshold: Threshold for determining position changes

    Returns:
        Dictionary with performance metrics
    """
    trades_df = get_model_trades(strategy_result, ohlc, position_threshold)

    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'total_pnl': 0.0,
            'profit_factor': 0.0,
            'strategy_name': strategy_result.metadata.get('strategy_name', 'unknown')
        }

    # Calculate performance metrics
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    total_wins = len(winning_trades)
    total_losses = len(losing_trades)
    total_trades = len(trades_df)

    win_rate = total_wins / total_trades if total_trades > 0 else 0.0
    avg_win = winning_trades['pnl'].mean() if total_wins > 0 else 0.0
    avg_loss = losing_trades['pnl'].mean() if total_losses > 0 else 0.0
    payoff_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')

    gross_profit = winning_trades['pnl'].sum() if total_wins > 0 else 0.0
    gross_loss = abs(losing_trades['pnl'].sum()) if total_losses > 0 else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    simple_returns = ohlc['close'].pct_change()
    log_returns = np.log(ohlc['close']).diff()
    total_log_return = log_returns.sum()
    trade_returns = trades_df['return_pct'] if not trades_df.empty else pd.Series([])

    # Calculate volatility (standard deviation of returns)
    volatility = simple_returns.std() if len(simple_returns) > 0 else 0.0

    # Calculate period-based strategy returns (aligned signal × log returns)
    signal = strategy_result.signal
    aligned_signal = signal.shift(1).fillna(0)

    # Align log_returns with signal index
    log_returns_aligned = log_returns.reindex(aligned_signal.index).fillna(0)
    strat_returns = aligned_signal * log_returns_aligned

    # Kelly Criterion: (win_rate / abs(avg_loss)) - ((1 - win_rate) / abs(avg_win))
    # This is the conventional Kelly formula for trading
    if total_trades > 0 and avg_win != 0 and avg_loss != 0:
        kelly_fraction = (win_rate / abs(avg_loss)) - ((1 - win_rate) / abs(avg_win))
    else:
        kelly_fraction = 0.0

    # Sharpe ratio - period-based (annualized)
    # Uses daily strategy returns, includes flat periods
    strat_returns_std = strat_returns.std() if len(strat_returns) > 0 else 0.0
    if strat_returns_std > 0:
        daily_sharpe = strat_returns.mean() / strat_returns_std
        sharpe_ratio = daily_sharpe * np.sqrt(252)  # Annualize assuming 252 trading days
    else:
        sharpe_ratio = 0.0

    # Calculate holding time statistics
    avg_holding_time_hours = trades_df['holding_time_hours'].mean() if 'holding_time_hours' in trades_df.columns else 0.0
    min_holding_time_hours = trades_df['holding_time_hours'].min() if 'holding_time_hours' in trades_df.columns else 0.0
    max_holding_time_hours = trades_df['holding_time_hours'].max() if 'holding_time_hours' in trades_df.columns else 0.0

    # Calculate direction-specific metrics
    long_trades = trades_df[trades_df['direction'] == 'LONG'] if 'direction' in trades_df.columns else pd.DataFrame()
    short_trades = trades_df[trades_df['direction'] == 'SHORT'] if 'direction' in trades_df.columns else pd.DataFrame()

    long_win_rate = len(long_trades[long_trades['result'] == 'WIN']) / len(long_trades) if len(long_trades) > 0 else 0.0
    short_win_rate = len(short_trades[short_trades['result'] == 'WIN']) / len(short_trades) if len(short_trades) > 0 else 0.0

    long_avg_pnl = long_trades['pnl'].mean() if len(long_trades) > 0 else 0.0
    short_avg_pnl = short_trades['pnl'].mean() if len(short_trades) > 0 else 0.0

    # Calculate drawdown metrics using equity-based approach
    # Starting capital for equity calculation (conventional backtesting amount)
    starting_capital = 10000.0

    # Build cumulative PnL series from trades
    if not trades_df.empty:
        cum_pnl_series = pd.Series(0.0, index=ohlc.index)
        cumulative_pnl = 0.0
        for _, trade in trades_df.iterrows():
            cumulative_pnl += trade['pnl']
            # Apply PnL from exit time onwards
            cum_pnl_series.loc[trade['exit_time']:] = cumulative_pnl

        # Calculate equity curve (starting capital + cumulative PnL)
        equity_curve = starting_capital + cum_pnl_series

        # Calculate drawdown from peak equity as percentage
        running_max_equity = equity_curve.cummax()
        drawdown_pct = ((running_max_equity - equity_curve) / running_max_equity) * 100
        max_drawdown = drawdown_pct.max() if len(drawdown_pct) > 0 else 0.0
    else:
        drawdown_pct = pd.Series(0.0, index=ohlc.index)
        max_drawdown = 0.0

    # Calculate drawdown duration (number of periods in drawdown)
    max_drawdown_duration = 0
    current_drawdown_duration = 0
    for dd in drawdown_pct:
        if dd > 0:
            current_drawdown_duration += 1
            max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
        else:
            current_drawdown_duration = 0

    # Sortino ratio - period-based (annualized)
    # Uses downside deviation of strategy returns (only negative returns)
    downside_returns = strat_returns[strat_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    if downside_std > 0:
        daily_sortino = strat_returns.mean() / downside_std
        sortino_ratio = daily_sortino * np.sqrt(252)  # Annualize
    else:
        sortino_ratio = 0.0

    # Calmar ratio - annualized return / max drawdown
    # Calculate total period in years
    if len(ohlc) > 1:
        total_days = (ohlc.index[-1] - ohlc.index[0]).days
        years = total_days / 365.25
        total_return = strat_returns.sum()
        annualized_return = total_return / years if years > 0 else 0.0
    else:
        annualized_return = 0.0

    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = 0.0

    # Calculate trade expectancy
    trade_expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Calculate largest win and loss
    largest_win = winning_trades['pnl'].max() if total_wins > 0 else 0.0
    largest_loss = losing_trades['pnl'].min() if total_losses > 0 else 0.0

    wealth_levels = 1 + trade_returns
    positive_wealth = wealth_levels[wealth_levels > 0]

    def _log_utility():
        if len(positive_wealth) == 0:
          return float('-inf')
        eu_log = np.mean(np.log(positive_wealth))
        return eu_log

    def _crra(gamma=2.0):
        if gamma == 1.0:
            return _log_utility()
        if len(positive_wealth) == 0:
          return float('-inf')
        utilities = (positive_wealth**(1 - gamma) - 1) / (1 - gamma)
        eu_crra = np.mean(utilities)
        return eu_crra
        

    metrics = {
        'Strategy name': strategy_result.metadata.get('strategy_name', 'unknown'),
        'Total trades': total_trades,
        'Winning trades': total_wins,
        'Losing trades': total_losses,
        'Win rate': win_rate,
        'Avg win': avg_win,
        'Avg loss': avg_loss,
        'Payoff ratio': payoff_ratio,
        'Total pnl': trades_df['pnl'].sum(),
        'Average pnl': trades_df['pnl'].mean(),
        'Profit factor': profit_factor,
        'Trade expectancy': trade_expectancy,
        'Largest win': largest_win,
        'Largest loss': largest_loss,
        'Avg returns': strat_returns.mean(),  # Strategy returns, not buy-and-hold
        'Volatility': strat_returns.std(),  # Strategy volatility, not market volatility
        'Kelly': kelly_fraction,
        'Sharpe ratio': sharpe_ratio,
        'Sortino ratio': sortino_ratio,
        'Calmar ratio': calmar_ratio,
        'Max drawdown': max_drawdown,
        'Max drawdown duration (periods)': max_drawdown_duration,
        'Logarithmic EU': _log_utility(),
        'CRRA EU': _crra(),
        'Sample Size': total_trades,
        'Adequate sample size': True if total_trades >= 30 else False,  # Statistical adequacy threshold
        'Max consecutive wins': _get_max_consecutive(trades_df['pnl'] > 0),
        'Max consecutive losses': _get_max_consecutive(trades_df['pnl'] < 0),
        'Avg hold (hours)': avg_holding_time_hours,
        'Min hold (hours)': min_holding_time_hours,
        'Max hold (hours)': max_holding_time_hours,
        'Long trades': len(long_trades),
        'Short trades': len(short_trades),
        'Long win rate': long_win_rate,
        'Short win rate': short_win_rate,
        'Long average pnl': long_avg_pnl,
        'Short average pnl': short_avg_pnl,
    }
    
    return metrics


def _get_max_consecutive(condition_series: pd.Series) -> int:
    """Helper function to calculate maximum consecutive occurrences."""
    if condition_series.empty:
        return 0

    max_consecutive = 0
    current_consecutive = 0

    for condition in condition_series:
        if condition:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def _calculate_holding_time_hours(holding_time: pd.Timedelta) -> float:
    """Helper function to convert pandas Timedelta to hours."""
    if pd.isna(holding_time):
        return 0.0
    return holding_time.total_seconds() / 3600.0

def print_metrics(metrics: Dict[str, Any]) -> None:
    """Helper function to print metrics in a formatted way."""
    for metric_name, value in metrics.items():
        if metric_name != 'trades_dataframe':
            if isinstance(value, float):
                print(f"{metric_name}: {value:.4f}")
            else:
                print(f"{metric_name}: {value}")