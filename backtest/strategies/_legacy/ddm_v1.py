import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

try:
    from backtest.validation.strategy_base import TradingStrategy, WalkForwardStrategy, StrategyResult
except ImportError:
    from backtest.validation.strategy_base import TradingStrategy, WalkForwardStrategy, StrategyResult


class DDMStrategy(TradingStrategy, WalkForwardStrategy):

    def __init__(self, name: str = "DDM Strategy"):
        super().__init__(name)

    def generate_signal(self, ohlc: pd.DataFrame, **params) -> StrategyResult:
        """
        Generate DDM trading signals based on basis (futures - spot spread).

        Strategy logic:
        - Computes basis = (futures - spot) / spot
        - Calculates upper/lower percentile thresholds on historical basis
        - Signals short (-1) when basis > upper threshold (futures overpriced)
        - Signals long (+1) when basis < lower threshold (futures underpriced)

        Args:
            ohlc: DataFrame with columns ['spot', 'futures', 'open', 'high', 'low', 'close']
            params:
                theta: percentile threshold for basis deviation (default=90)
                    - Higher theta = more extreme deviations needed for signal
                    - theta=90 means signal when basis > 90th percentile or < 10th percentile
                n_days: (ONLY for real contracts) number of days before expiry to filter
                expiry: (ONLY for real contracts) expiry date or None for continuous futures
                relative_mode: (ONLY for real contracts) if True, use index-based cutoff
                threshold_data: (Optional) Pre-computed basis data for threshold calculation
                    - If provided, uses this data instead of ohlc for computing thresholds
                    - This prevents look-ahead bias in walk-forward testing

        Returns:
            StrategyResult with signal (+1=long, -1=short, 0=neutral)
        """
        theta = params.get("theta", 90) #A value of theta needs to be determined from backtesting, at the moment it is set to 90% (i.e. 10% will trigger a trade)
        n_days = params.get("n_days", 5)
        expiry = params.get("expiry", None)
        relative_mode = params.get("relative_mode", False)
        threshold_data_override = params.get("threshold_data", None)

        # --- compute basis (use copy to avoid SettingWithCopyWarning)
        ohlc = ohlc.copy()
        ohlc["basis"] = (ohlc["futures"] - ohlc["spot"]) / ohlc["spot"]

        # --- Determine threshold data source
        if threshold_data_override is not None:
            # Use pre-computed threshold data (for walk-forward to prevent look-ahead)
            threshold_data = threshold_data_override
        elif expiry is None:
            # Continuous futures mode: use full history for thresholds
            threshold_data = ohlc["basis"]
        else:
            # Real contract mode: apply maturity filter for threshold calculation
            df_filtered = self._filter_near_maturity(ohlc, expiry, n_days, relative_mode)
            threshold_data = df_filtered["basis"]

        # --- compute thresholds
        upper = np.nanpercentile(threshold_data, theta)
        lower = np.nanpercentile(threshold_data, 100 - theta)

        # --- generate trading signals on full dataset
        signal = pd.Series(0, index=ohlc.index)
        signal.loc[ohlc["basis"] > upper] = -1  # Futures overpriced → short
        signal.loc[ohlc["basis"] < lower] = 1   # Futures underpriced → long

        metadata = {
            "theta": theta,
            "upper_threshold": upper,
            "lower_threshold": lower,
            "basis": ohlc["basis"],
            "strategy_name": self.name
        }

        return StrategyResult(signal=signal, metadata=metadata)

    def _filter_near_maturity(self, df: pd.DataFrame, expiry=None, n_days: int = 5, relative_mode=False):
        """Filter data to only include last n_days before expiry."""
        if relative_mode or expiry is None:
            # Index-based cutoff for Monte Carlo
            cutoff_index = int(len(df) * (1 - n_days / len(df)))
            return df.iloc[cutoff_index:]
        else:
            # Datetime-based cutoff for real contracts
            if isinstance(expiry, str):
                expiry_date = datetime.strptime(expiry, "%Y%m").replace(tzinfo=timezone.utc)
            elif isinstance(expiry, datetime):
                expiry_date = expiry
            else:
                raise TypeError("expiry must be str 'YYYYMM' or datetime")

            # Match timezone awareness
            if df.index.tz is not None:
                expiry_date = expiry_date.replace(tzinfo=df.index.tz)
            cutoff = expiry_date - timedelta(days=n_days)
            return df[df.index >= cutoff]

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Define parameter search space for optimisation.

        For continuous futures (expiry=None), only theta matters.
        For real contracts, theta and n_days both matter.
        """
        return {
            "theta": [80, 85, 90, 95],
            # Note: n_days only used when expiry is specified (real contracts)
            # For continuous futures, this parameter is ignored
        }

    def walk_forward_signal(self, ohlc: pd.DataFrame, train_lookback: int = 252*2, train_step: int = 30) -> pd.Series:
        """
        Walk-forward signal generation (periodic reoptimisation).

        Fixed to prevent look-ahead bias by computing thresholds ONLY on training data.
        """
        n = len(ohlc)
        wf_signal = np.full(n, np.nan)

        # Pre-compute basis for the entire dataset
        ohlc = ohlc.copy()
        ohlc["basis"] = (ohlc["futures"] - ohlc["spot"]) / ohlc["spot"]

        next_train = train_lookback
        best_params = None
        upper_threshold = None
        lower_threshold = None

        for i in range(next_train, n):
            # Retrain when we hit the next training point
            if i == next_train:
                train_data = ohlc.iloc[i - train_lookback:i]
                opt_result = self.optimise(train_data)
                best_params = opt_result.best_params

                # Compute thresholds ONLY on training data basis
                train_basis = train_data["basis"]
                theta = best_params.get("theta", 90)
                upper_threshold = np.nanpercentile(train_basis, theta)
                lower_threshold = np.nanpercentile(train_basis, 100 - theta)

                next_train += train_step

            # Generate signal at current bar using training thresholds
            current_basis = ohlc["basis"].iloc[i]
            if current_basis > upper_threshold:
                wf_signal[i] = -1  # Short
            elif current_basis < lower_threshold:
                wf_signal[i] = 1   # Long
            else:
                wf_signal[i] = 0   # Neutral

        return pd.Series(wf_signal, index=ohlc.index)
