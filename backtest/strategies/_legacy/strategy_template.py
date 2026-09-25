"""
Template for implementing new trading strategies with standardised interface.
Copy this file and implement your custom strategy following the examples.
Please try to follow the template it makes my life 1000x easier.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
try:
    from backtest.validation.strategy_base import TradingStrategy, WalkForwardStrategy, StrategyResult
except ImportError:
    from backtest.validation.strategy_base import TradingStrategy, WalkForwardStrategy, StrategyResult


class CustomStrategy(TradingStrategy, WalkForwardStrategy):
    """Template for implementing custom trading strategies."""

    def __init__(self, name: str = "Custom Strategy"):
        super().__init__(name)

    def generate_signal(self, ohlc: pd.DataFrame, **params) -> StrategyResult:
        """
        Generate trading signals.

        Args:
            ohlc: DataFrame with OHLC data (must have 'open', 'high', 'low', 'close' columns)
            **params: Strategy-specific parameters

        Returns:
            StrategyResult with trading signals (+1 for long, -1 for short, 0 for neutral)

        Example implementation:
        """
        # Example: Basic moving average crossover
        fast_ma = params.get('fast_ma', 10)
        slow_ma = params.get('slow_ma', 20)

        fast_sma = ohlc['close'].rolling(fast_ma).mean()
        slow_sma = ohlc['close'].rolling(slow_ma).mean()

        signal = pd.Series(0, index=ohlc.index)
        signal[fast_sma > slow_sma] = 1   # Long when fast MA above slow MA
        signal[fast_sma < slow_sma] = -1  # Short when fast MA below slow MA

        metadata = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'fast_sma': fast_sma,
            'slow_sma': slow_sma,
            'strategy_name': self.name
        }

        return StrategyResult(signal=signal, metadata=metadata)

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Define parameter space for optimisation.

        Returns:
            Dictionary mapping parameter names to lists of possible values

        Example:
        """
        return {
            'fast_ma': list(range(5, 50, 5)),    # Fast MA from 5 to 45 in steps of 5
            'slow_ma': list(range(20, 100, 10))  # Slow MA from 20 to 90 in steps of 10
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

        Example implementation:
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
                best_params = opt_result.best_params

                # Generate signal for entire dataset with optimised parameters
                tmp_signal = self.generate_signal(ohlc, **best_params).signal
                next_train += train_step

            wf_signal[i] = tmp_signal.iloc[i]

        return pd.Series(wf_signal, index=ohlc.index)