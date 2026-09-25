import pandas as pd
import numpy as np
from typing import Dict, Any, List

try:
    from backtest.validation.strategy_base import TradingStrategy, WalkForwardStrategy, StrategyResult
except ImportError:
    from backtest.validation.strategy_base import TradingStrategy, WalkForwardStrategy, StrategyResult


class DDMv2Strategy(TradingStrategy, WalkForwardStrategy):
    """
    DDM v2: Z-score normalised version.

    Key difference from v1:
    - v1: Uses percentile thresholds on raw basis values (sensitive to regime shifts)
    - v2: Uses z-score normalisation with rolling window (adapts to regime shifts)

    Instead of asking "is basis > 90th percentile historically?", v2 asks:
    "is basis > 2 standard deviations from recent mean?"

    This makes the strategy robust to non-stationarity in basis levels.
    """

    def __init__(self, name: str = "DDM v2 Strategy"):
        super().__init__(name)

    def generate_signal(self, ohlc: pd.DataFrame, **params) -> StrategyResult:
        """
        Generate DDM v2 trading signals using z-score normalisation.

        Strategy logic:
        - Computes basis = (futures - spot) / spot
        - Calculates rolling mean and std of basis
        - Computes z-score = (basis - rolling_mean) / rolling_std
        - Signals short (-1) when z-score > threshold (futures overpriced)
        - Signals long (+1) when z-score < -threshold (futures underpriced)

        Args:
            ohlc: DataFrame with columns ['spot', 'futures', 'open', 'high', 'low', 'close']
            params:
                z_threshold: number of standard deviations for signal (default=2.0)
                    - Higher threshold = more extreme deviations needed
                    - z_threshold=2.0 means signal when |z-score| > 2
                lookback: rolling window size for mean/std calculation (default=252)
                    - 252 = 1 year for daily data
                    - Larger = slower adaptation, smaller = faster but noisier
                min_periods: minimum periods needed for calculation (default=50)

        Returns:
            StrategyResult with signal (+1=long, -1=short, 0=neutral)
        """
        z_threshold = params.get("z_threshold", 2.0)
        lookback = params.get("lookback", 252)
        min_periods = params.get("min_periods", 50)

        # Compute basis
        ohlc = ohlc.copy()
        ohlc["basis"] = (ohlc["futures"] - ohlc["spot"]) / ohlc["spot"]

        # Calculate rolling statistics
        # IMPORTANT: Shift by 1 to prevent look-ahead bias
        # At bar t, we use rolling stats from bars [t-lookback, t-1], NOT including bar t
        rolling_mean = ohlc["basis"].rolling(window=lookback, min_periods=min_periods).mean().shift(1)
        rolling_std = ohlc["basis"].rolling(window=lookback, min_periods=min_periods).std().shift(1)

        # Compute z-score using yesterday's rolling stats
        z_score = (ohlc["basis"] - rolling_mean) / rolling_std

        # Generate trading signals
        signal = pd.Series(0, index=ohlc.index)
        signal.loc[z_score > z_threshold] = -1   # Futures overpriced → short
        signal.loc[z_score < -z_threshold] = 1   # Futures underpriced → long

        metadata = {
            "z_threshold": z_threshold,
            "lookback": lookback,
            "basis": ohlc["basis"],
            "z_score": z_score,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "strategy_name": self.name
        }

        return StrategyResult(signal=signal, metadata=metadata)

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Define parameter search space for optimisation.

        z_threshold: How many standard deviations constitute a signal
        lookback: Window size for rolling statistics
        """
        return {
            "z_threshold": [1.5, 2.0, 2.5, 3.0],
            "lookback": [126, 252, 504],  # 6 months, 1 year, 2 years
        }

    def walk_forward_signal(self, ohlc: pd.DataFrame, train_lookback: int = 252*2, train_step: int = 30) -> pd.Series:
        """
        Walk-forward signal generation with periodic reoptimisation.

        Note: Z-score approach is inherently forward-looking (uses rolling window),
        but doesn't have look-ahead bias because it only uses data up to current bar.
        """
        n = len(ohlc)
        wf_signal = np.full(n, np.nan)

        # Pre-compute basis
        ohlc = ohlc.copy()
        ohlc["basis"] = (ohlc["futures"] - ohlc["spot"]) / ohlc["spot"]

        next_train = train_lookback
        best_params = None

        for i in range(train_lookback, n):
            # Retrain when we hit the next training point
            if i == next_train:
                train_data = ohlc.iloc[i - train_lookback:i]
                opt_result = self.optimise(train_data)
                best_params = opt_result.best_params
                next_train += train_step

            # Generate signal at current bar
            # Use only data up to current bar (no look-ahead)
            if best_params is not None:
                current_slice = ohlc.iloc[:i+1]
                result = self.generate_signal(current_slice, **best_params)
                wf_signal[i] = result.signal.iloc[-1]

        return pd.Series(wf_signal, index=ohlc.index)
