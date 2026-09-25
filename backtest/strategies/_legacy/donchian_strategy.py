"""
Donchian Breakout Strategy implementation using standardised interface.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
try:
    from backtest.validation.strategy_base import TradingStrategy, WalkForwardStrategy, StrategyResult
except ImportError:
    from backtest.validation.strategy_base import TradingStrategy, WalkForwardStrategy, StrategyResult # type: ignore


class DonchianStrategy(TradingStrategy, WalkForwardStrategy):
    """Donchian Breakout strategy implementation."""

    def __init__(self):
        super().__init__("Donchian Breakout")

    def generate_signal(self, ohlc: pd.DataFrame, lookback: int = 20) -> StrategyResult:
        """
        Generate Donchian breakout signals.

        Args:
            ohlc: DataFrame with OHLC data
            lookback: Lookback period for high/low calculation

        Returns:
            StrategyResult with trading signals
        """
        ohlc = self._normalise_ohlc(ohlc)

        upper = ohlc['close'].rolling(lookback - 1).max().shift(1)
        lower = ohlc['close'].rolling(lookback - 1).min().shift(1)

        signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
        signal.loc[ohlc['close'] > upper] = 1
        signal.loc[ohlc['close'] < lower] = -1
        signal = signal.ffill()

        metadata = {
            'lookback': lookback,
            'upper_band': upper,
            'lower_band': lower,
            'strategy_name': self.name
        }

        return StrategyResult(signal=signal, metadata=metadata)

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Define parameter space for optimisation."""
        return {
            'lookback': list(range(12, 169))  # Original range from donchian.py
        }

    def walk_forward_signal(self, ohlc: pd.DataFrame, train_lookback: int = 24*365*4,
                           train_step: int = 24*30) -> pd.Series:
        """
        Generate walk-forward signals with periodic reoptimisation.

        Args:
            ohlc: DataFrame with OHLC data
            train_lookback: Number of periods for training window
            train_step: Number of periods between retraining

        Returns:
            Series with walk-forward signals
        """
        n = len(ohlc)
        wf_signal = np.full(n, np.nan)
        tmp_signal = None

        next_train = train_lookback
        for i in range(next_train, n):
            if i == next_train:
                # Optimise on training window
                train_data = ohlc.iloc[i-train_lookback:i]
                opt_result = self.optimise(train_data)
                best_lookback = opt_result.best_params['lookback']

                # Generate signal for entire dataset with optimised parameters
                tmp_signal = self.generate_signal(ohlc, lookback=best_lookback).signal
                next_train += train_step

            wf_signal[i] = tmp_signal.iloc[i]

        return pd.Series(wf_signal, index=ohlc.index)

if __name__ == '__main__':
    """Example usage of the standardised Donchian strategy."""
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Run strategy demo')
    parser.add_argument('--data-file', '-d', required=True, help='Path to data file (CSV, Parquet, JSON, Excel)')
    parser.add_argument('--start-year', type=int, help='Start year for filtering')
    parser.add_argument('--end-year', type=int, help='End year for filtering')
    args = parser.parse_args()

    if args.data_file.endswith('.csv'):
        df = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
    elif args.data_file.endswith('.parquet') or args.data_file.endswith('.pq'):
        df = pd.read_parquet(args.data_file)
    elif args.data_file.endswith('.json'):
        df = pd.read_json(args.data_file, orient='index')
    elif args.data_file.endswith(('.xlsx', '.xls')):       
        df = pd.read_excel(args.data_file, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {args.data_file}")
    
    df.index = pd.to_datetime(df.index)

    if args.start_year or args.end_year:
        start_year = args.start_year or df.index.year.min()
        end_year = args.end_year or df.index.year.max()    
        df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

    # Create strategy instance
    strategy = DonchianStrategy()

    # Optimise parameters
    opt_result = strategy.optimise(df)
    print(f"Best lookback: {opt_result.best_params['lookback']}")
    print(f"Best profit factor: {opt_result.best_score:.4f}")

    # Generate signals with optimised parameters
    result = strategy.generate_signal(df, **opt_result.best_params)

    # Calculate returns
    price_returns = np.log(df['close']).diff().shift(-1)
    strategy_returns = result.get_returns(price_returns)

    # Plot results
    plt.style.use("dark_background")
    strategy_returns.cumsum().plot(color='red')
    plt.title("Standardised Donchian Breakout Strategy")
    plt.ylabel('Cumulative Log Return')
    plt.show()