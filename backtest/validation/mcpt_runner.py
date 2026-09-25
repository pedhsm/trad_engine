"""
Unified MCPT testing framework.
Provides MCPTRunner class and standalone functions for running
in-sample and walk-forward Monte Carlo Permutation Tests.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .strategy_base import TradingStrategy, MCPTTester

class MCPTRunner:
    """High-level interface for running MCPT tests on trading strategies."""

    def __init__(self, strategy: TradingStrategy):
        """
        Initialise MCPT runner.

        Args:
            strategy: Trading strategy to test
        """
        self.strategy = strategy
        self.tester = MCPTTester(strategy)
        self.data = None



    def run_insample_mcpt(self, n_permutations: int = 1000, perm_kwargs: dict = None, **strategy_params) -> Dict[str, Any]:
        """
        Run in-sample MCPT with default or provided parameters.

        Args:
            n_permutations: Number of permutations to test
            perm_kwargs: Additional kwargs for permutation (e.g., {'permutation_mode': 'row'})
            **strategy_params: Strategy parameters to use (if empty, uses strategy defaults)

        Returns:
            Dictionary with test results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(f"Running in-sample MCPT for {self.strategy.name}")

        # Use provided parameters or defaults (no optimisation to avoid data leakage)
        if strategy_params:
            print(f"Using provided parameters: {strategy_params}")
        else:
            print("Using strategy default parameters (no optimisation)")
            strategy_params = {}

        # Run MCPT
        print(f"Testing {n_permutations} permutations...")
        if perm_kwargs is None:
            perm_kwargs = {}
        results = self.tester.run_insample_test(
            self.data, n_permutations, perm_kwargs=perm_kwargs, **strategy_params
        )

        print(f"In-sample MCPT - Cumulative Log Return: {results['real_cumulative_return']:.4f}, P-Value: {results['p_value']:.4f}")
        return results

    def run_walkforward_mcpt(self, n_permutations: int = 200, train_lookback: int = 24*365*4,
                           train_step: int = 24*30, perm_kwargs: dict = None) -> Dict[str, Any]:
        """
        Run walk-forward MCPT.

        Args:
            n_permutations: Number of permutations to test
            train_lookback: Training window size
            train_step: Retraining frequency
            perm_kwargs: Additional kwargs for permutation (e.g., {'permutation_mode': 'row'})

        Returns:
            Dictionary with test results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(f"Running walk-forward MCPT for {self.strategy.name}")
        print(f"Training window: {train_lookback} periods, Step: {train_step} periods")

        # Run MCPT
        print(f"Testing {n_permutations} permutations...")
        if perm_kwargs is None:
            perm_kwargs = {}
        results = self.tester.run_walkforward_test(
            self.data, n_permutations, train_lookback, train_step, perm_kwargs=perm_kwargs
        )

        print(f"Walk-forward MCPT - Cumulative Log Return: {results['real_cumulative_return']:.4f}, P-Value: {results['p_value']:.4f}")
        return results

    def plot_results(self, results: Dict[str, Any], test_type: str = "MCPT"):
        """
        Plot MCPT results with both profit factor histogram and cumulative returns.

        Args:
            results: Results from MCPT test
            test_type: Type of test for plot title
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        plt.style.use('dark_background')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Cumulative Log Returns Histogram
        cum_returns_series = pd.Series(results['permuted_cumulative_returns'])

        cum_returns_series.hist(ax=ax1, color='blue', label='Permutations', alpha=0.7, bins=50)

        real_cum_return = results['real_cumulative_return']
        ax1.axvline(real_cum_return, color='red',
                   label=f'Real ({real_cum_return:.3f})', linewidth=2)

        ax1.set_xlabel("Cumulative Log Returns")
        ax1.set_title(f'Cumulative Returns Distribution - P-Value: {results["p_value"]:.4f}')
        ax1.grid(False)
        ax1.legend()

        # Cumulative Returns Time Series
        if 'real_cum_returns' in results and 'permuted_cum_returns' in results:
            # Plot permutation curves
            perm_sample = results['permuted_cum_returns']
            for i, perm_cum_ret in enumerate(perm_sample):
                dates = mdates.date2num(pd.to_datetime(perm_cum_ret.index))
                ax2.plot(dates, perm_cum_ret.values,
                        color='blue', alpha=0.3, linewidth=0.5)

            # Plot real cumulative returns
            real_cum_ret = results['real_cum_returns']
            dates = mdates.date2num(pd.to_datetime(real_cum_ret.index))
            ax2.plot(dates, real_cum_ret.values,
                    color='red', linewidth=2, label='Real Strategy')

            # Format x-axis as dates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

            ax2.set_xlabel("Time")
            ax2.set_ylabel("Cumulative Log Returns")
            ax2.set_title("Cumulative Returns: Real vs Permutations")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        plt.suptitle(f'{self.strategy.name} {test_type}', fontsize=16)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Standalone functions (consolidated from insample_mcpt.py / walkforward_mcpt.py)
# ---------------------------------------------------------------------------

def _filter_data(ohlc: pd.DataFrame,
                 start_date: Optional[str],
                 end_date: Optional[str]) -> pd.DataFrame:
    """Filter OHLC data by date range."""
    if start_date is not None or end_date is not None:
        mask = pd.Series(True, index=ohlc.index)
        if start_date is not None:
            mask &= (ohlc.index >= pd.to_datetime(start_date))
        if end_date is not None:
            mask &= (ohlc.index < pd.to_datetime(end_date))
        return ohlc[mask]
    return ohlc


def run_insample(ohlc: pd.DataFrame,
                 strategy_name: str,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 n_permutations: int = 1000,
                 permutation_mode: str = 'bar',
                 block_length: Optional[int] = None):
    """
    Run configurable in-sample MCPT.

    Args:
        ohlc: DataFrame with OHLC data
        strategy_name: Name of strategy to test
        start_date: Start date for data filtering (YYYY-MM-DD format)
        end_date: End date for data filtering (YYYY-MM-DD format)
        n_permutations: Number of permutations to test
        permutation_mode: Permutation mode ('bar', 'row', 'ar1', or 'block')
        block_length: Block length for block bootstrap (only used if permutation_mode='block')
    """
    # Lazy import to avoid circular dependency with engine
    from backtest.engine.config import get_strategy_by_name

    print(f"Strategy: {strategy_name}")
    print(f"Permutations: {n_permutations}")

    try:
        strategy = get_strategy_by_name(strategy_name)
        runner = MCPTRunner(strategy)
        runner.data = _filter_data(ohlc, start_date, end_date)

        print(f"Loaded data: {len(runner.data)} rows from {runner.data.index[0]} to {runner.data.index[-1]}")

        perm_kwargs = {'permutation_mode': permutation_mode}
        if block_length is not None:
            perm_kwargs['block_length'] = block_length

        results = runner.run_insample_mcpt(n_permutations=n_permutations,
                                            perm_kwargs=perm_kwargs)

        runner.plot_results(results, "In-Sample MCPT")
  
        print("\nResults:")
        print(f"Strategy: {strategy.name}")

        print(f"Time Period: {start_date}-{end_date}")
        print(f"Data Points: {len(runner.data)}")
        print(f"Optimised Parameters: {results['strategy_params']}")
        print(f"Real Cumulative Log Return: {results['real_cumulative_return']:.6f}")
        print(f"Mean Permuted Cum Log Return: {np.mean(results['permuted_cumulative_returns']):.6f}")
        print(f"Std Permuted Cum Log Return: {np.std(results['permuted_cumulative_returns']):.6f}")
        print(f"Real Profit Factor (bar-by-bar): {results['real_profit_factor']:.6f}")
        print(f"P-Value: {results['p_value']:.6f}")

        return results

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def run_walkforward(ohlc: pd.DataFrame,
                    strategy_name: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    n_permutations: int = 200,
                    train_lookback: Optional[int] = None,
                    train_step: Optional[int] = None,
                    permutation_mode: str = 'bar',
                    block_length: Optional[int] = None):
    """
    Run configurable walk-forward MCPT.

    Args:
        ohlc: DataFrame with OHLC data
        strategy_name: Name of strategy to test
        start_date: Start date for data filtering (YYYY-MM-DD format)
        end_date: End date for data filtering (YYYY-MM-DD format)
        n_permutations: Number of permutations to test
        train_lookback: Training window size in periods
        train_step: Retraining frequency in periods
        permutation_mode: Permutation mode ('bar', 'row', 'ar1', or 'block')
        block_length: Block length for block bootstrap (only used if permutation_mode='block')
    """
    # Lazy import to avoid circular dependency with engine
    from backtest.engine.config import get_strategy_by_name

    try:
        strategy = get_strategy_by_name(strategy_name)
        runner = MCPTRunner(strategy)
        runner.data = _filter_data(ohlc, start_date, end_date)

        print(f"Loaded data: {len(runner.data)} rows from {runner.data.index[0]} to {runner.data.index[-1]}")
        data_size = len(runner.data)
        if train_lookback is None:
            train_lookback = min(30000, max(20000, data_size // 3))
        if train_step is None:
            train_step = max(5000, train_lookback // 6)

        print(f"Using training window: {train_lookback} periods, step: {train_step} periods")

        perm_kwargs = {'permutation_mode': permutation_mode}
        if block_length is not None:
            perm_kwargs['block_length'] = block_length

        results = runner.run_walkforward_mcpt(
            n_permutations=n_permutations,
            train_lookback=train_lookback,
            train_step=train_step,
            perm_kwargs=perm_kwargs
        )

        runner.plot_results(results, "Walk-Forward MCPT")


        print("\nResults:")
        print(f"Strategy: {strategy.name}")

        print(f"Time Period: {start_date}-{end_date}")
        print(f"Data Points: {len(runner.data)}")
        print(f"Training Window: {results['train_lookback']} periods")
        print(f"Retraining Step: {results['train_step']} periods")
        print(f"Real Cumulative Log Return: {results['real_cumulative_return']:.6f}")
        print(f"Mean Permuted Cum Log Return: {np.mean(results['permuted_cumulative_returns']):.6f}")
        print(f"Std Permuted Cum Log Return: {np.std(results['permuted_cumulative_returns']):.6f}")
        print(f"Real Profit Factor (bar-by-bar): {results['real_profit_factor']:.6f}")
        print(f"P-Value: {results['p_value']:.6f}")

        return results

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    """Demo usage of MCPTRunner framework."""
    print("MCPTRunner Framework")
    print("Use this with the strategy registry and backtest.py for testing.")
    print("Example:")
    print("  from backtest.engine.config import get_strategy_by_name")
    print("  strategy = get_strategy_by_name('donchian')")
    print("  runner = MCPTRunner(strategy)")
    print("  runner.data = ohlc")
    print("  results = runner.run_insample_mcpt()")