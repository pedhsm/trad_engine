"""
Walk-forward validation (no permutations).
Pure rolling window walk-forward.
"""
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None
import argparse
from typing import Optional

try:
    from backtest.engine.config import get_strategy_by_name, list_strategies
    from backtest.validation.mcpt_runner import MCPTRunner
except ImportError:
    from backtest.engine.config import get_strategy_by_name, list_strategies


def main(ohlc: pd.DataFrame,
         strategy_name: str,
         start_date: Optional[str] = None,
         end_date: Optional[str] = None,
         train_lookback: Optional[int] = None,
         train_step: Optional[int] = None,
         plot: bool = True):
    """
    Args:
        strategy_name: Name of strategy to test
        start_date: Start date for data filtering (YYYY-MM-DD format)
        end_date: End date for data filtering (YYYY-MM-DD format)
        train_lookback: Training window size in periods
        train_step: Retraining frequency in periods
    """
    try:
        strategy = get_strategy_by_name(strategy_name)
        
        # We manually filter data since we removed load_data from MCPTRunner
        if start_date is not None or end_date is not None:
            mask = pd.Series(True, index=ohlc.index)
            if start_date is not None:
                start_dt = pd.to_datetime(start_date)
                mask &= (ohlc.index >= start_dt)
            if end_date is not None:
                end_dt = pd.to_datetime(end_date)
                mask &= (ohlc.index < end_dt)
            ohlc = ohlc[mask]
        
        data_size = len(ohlc)
        if train_lookback is None:
            train_lookback = min(30000, max(20000, data_size // 3))
        if train_step is None:
            train_step = max(5000, train_lookback // 6)

        print("\nWalk-Forward Configuration:")
        print(f"Training Window: {train_lookback} periods")
        print(f"Retraining Step: {train_step} periods")

        signal = strategy.walk_forward_signal(ohlc, train_lookback, train_step)

        # Pass the RAW signal: analyse_model_performance applies the one-bar shift
        # itself (the position decided at the close of t is held over t -> t+1).
        # Shifting here as well would delay every position by two bars.
        from backtest.validation.strategy_base import StrategyResult
        from backtest.engine.trade_metrics import analyse_model_performance
        strategy_result = StrategyResult(signal=signal, metadata={'strategy_name': strategy.name})
        metrics = analyse_model_performance(strategy_result, ohlc, position_threshold=0.0)

        # Same convention for the equity curve: position decided at t-1 times the
        # log return of (t-1, t].
        strategy_returns = signal.shift(1) * np.log(ohlc['close']).diff()
        cum_returns = strategy_returns.cumsum()

        print(f"\n{'='*60}")
        print("WALK-FORWARD RESULTS")
        print(f"{'='*60}")
        print(f"Strategy: {strategy.name}")

        print(f"Time Period: {start_date or ohlc.index[0]} to {end_date or ohlc.index[-1]}")
        print(f"Data Points: {len(ohlc)}")
        print(f"Training Window: {train_lookback} periods")
        print(f"Retraining Step: {train_step} periods")

        print(f"\n{'='*60}")
        print("PERFORMANCE METRICS")
        print(f"{'='*60}")

        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) > 1000 or abs(value) < 0.001:
                    print(f"{key}: {value:.6e}")
                else:
                    print(f"{key}: {value:.4f}")
            elif isinstance(value, bool):
                print(f"{key}: {'Yes' if value else 'No'}")
            else:
                print(f"{key}: {value}")

        if plot and plt is not None:
            plot_walkforward_results(strategy.name, metrics.get('Profit factor', 0.0), cum_returns)

        results = {
            'strategy_name': strategy.name,
            'metrics': metrics,
            'cum_returns': cum_returns,
            'signal': signal,
            'train_lookback': train_lookback,
            'train_step': train_step
        }

        return results

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please ensure the data file exists in the specified path.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_walkforward_results(strategy_name: str, profit_factor: float, cum_returns: pd.Series):
    """
    Plot walk-forward results (cumulative returns time series like MCPT but without permutations).

    Args:
        strategy_name: Name of the strategy
        profit_factor: Calculated profit factor
        cum_returns: Cumulative returns series
    """
    import matplotlib.dates as mdates

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    dates = mdates.date2num(pd.to_datetime(cum_returns.index))
    ax.plot(dates, cum_returns.values, color='red', linewidth=2, label='Strategy Returns')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Cumulative Log Returns', fontsize=12)
    ax.set_title(f'{strategy_name} Walk-Forward - PF: {profit_factor:.4f}', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Walk-Forward Validation (No Permutations)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument('--strategy', '-s',
                       help='Strategy name (registered in strategy registry)')
    parser.add_argument('--asset', '-a', default='SPY',
                       help='Asset to load from the data lake')
    parser.add_argument('--list-strategies', action='store_true',
                       help='List available strategies and exit')
    parser.add_argument('--start-date', type=str,
                       help='Start date for data filtering (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str,
                       help='End date for data filtering (YYYY-MM-DD format)')
    parser.add_argument('--train-lookback', type=int,
                       help='Training window size in periods (auto-adjusted if not specified)')
    parser.add_argument('--train-step', type=int,
                       help='Retraining frequency in periods (auto-adjusted if not specified)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.list_strategies:
        list_strategies()
        raise SystemExit(0)
    if not args.strategy:
        raise SystemExit("--strategy is required")

    from backtest.engine.backtest import DuckDBDataLoader
    data = DuckDBDataLoader.load_data(args.asset, start_date=args.start_date, end_date=args.end_date)
    results = main(
        ohlc=data,
        strategy_name=args.strategy,
        start_date=args.start_date,
        end_date=args.end_date,
        train_lookback=args.train_lookback,
        train_step=args.train_step
    )
