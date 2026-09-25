"""
Base classes and interfaces for standardised trading strategy implementation.
Provides plug-and-play architecture for future models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional (progress bars); degrade to a no-op shim
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is not None:
            return iterable

        class _NullBar:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def update(self, *a, **k):
                pass

        return _NullBar()


def _insample_signal_func(args):
    """Module-level signal function for in-sample MCPT."""
    os.environ['MCPT_SUBPROCESS'] = '1'
    ohlc, strategy, strategy_params = args
    result = strategy.generate_signal(ohlc, **strategy_params)
    # Use pre-calculated returns if available (for row permutation)
    # Otherwise calculate from close prices (for bar permutation)
    if '_precalc_returns' in ohlc.columns:
        price_returns = ohlc['_precalc_returns']
    else:
        # Calculate price returns without look-ahead bias
        # diff() gives return from t-1 to t: log(close[t]) - log(close[t-1])
        price_returns = np.log(ohlc['close']).diff()
    # Align signal: signal at bar t-1 predicts return from t-1 to t
    # So we shift signal forward by 1 to match with the return it's predicting
    aligned_signal = result.signal.shift(1)
    raw_returns = aligned_signal * price_returns
    spread_bps = strategy_params.get("spread_bps", 0.0) if isinstance(strategy_params, dict) else 0.0
    commission_bps = strategy_params.get("commission_bps", 0.0) if isinstance(strategy_params, dict) else 0.0
    if spread_bps > 0 or commission_bps > 0:
        trades = aligned_signal.diff().abs().fillna(0)
        costs = trades * ((spread_bps + commission_bps) / 10000.0)
        return raw_returns - costs
    return raw_returns


def _walkforward_signal_func(args):
    """Module-level signal function for walk-forward MCPT."""
    os.environ['MCPT_SUBPROCESS'] = '1'
    ohlc, strategy, train_lookback, train_step = args
    if '_precalc_returns' in ohlc.columns:
        price_returns = ohlc['_precalc_returns']
    else:
        price_returns = np.log(ohlc['close']).diff()
    signal = strategy.walk_forward_signal(ohlc, train_lookback, train_step)
    aligned_signal = signal.shift(1)
    return aligned_signal * price_returns


def _run_single_permutation(args):
    ohlc, signal_func, signal_args, perm_kwargs, seed = args
    from .bar_permute import get_permutation
    import numpy as np

    perm_kwargs_to_use = dict(perm_kwargs or {})
    perm_kwargs_to_use['seed'] = seed
    perm_ohlc = get_permutation(ohlc, **perm_kwargs_to_use)
    perm_returns = signal_func((perm_ohlc,) + signal_args)

    # Calculate cumulative return (primary test statistic)
    perm_cum_return = perm_returns.sum()

    # Calculate profit factor for reporting (mark-to-market based, not trade-based)
    wins = perm_returns[perm_returns > 0].sum()
    losses = perm_returns[perm_returns < 0].abs().sum()
    perm_pf = wins / losses if losses > 0 else np.inf if wins > 0 else 1.0

    perm_cum_returns = perm_returns.cumsum()
    return perm_cum_return, perm_pf, perm_cum_returns


@dataclass
class StrategyResult:
    """Standardised result from strategy execution."""
    signal: pd.Series
    metadata: Dict[str, Any]

    def get_returns(self, price_returns: pd.Series) -> pd.Series:
        """Calculate strategy returns given price returns."""
        return self.signal * price_returns

    def get_profit_factor(self, price_returns: pd.Series) -> float:
        """Calculate profit factor for the strategy."""
        strategy_returns = self.get_returns(price_returns)
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]

        if len(negative_returns) == 0:
            return np.inf if len(positive_returns) > 0 else 0.0

        return positive_returns.sum() / negative_returns.abs().sum()


@dataclass
class OptimisationResult:
    """Result from strategy parameter optimisation."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name

    def _normalise_ohlc(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Normalise OHLC DataFrame column names to lowercase."""
        ohlc = ohlc.copy()
        ohlc.columns = ohlc.columns.str.lower()

        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in ohlc.columns]
        if missing_cols:
            raise ValueError(f"Data must contain columns: {required_cols}. Missing: {missing_cols}")

        return ohlc

    @abstractmethod
    def generate_signal(self, ohlc: pd.DataFrame, **params) -> StrategyResult:
        """
        Generate trading signals for given OHLC data.

        Args:
            ohlc: DataFrame with OHLC data
            **params: Strategy-specific parameters

        Returns:
            StrategyResult containing signals and metadata
        """
        pass

    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Define the parameter space for optimisation.

        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        pass

    def optimise(self, ohlc: pd.DataFrame, score_func: str = 'profit_factor') -> OptimisationResult:
        """
        Optimise strategy parameters.

        Args:
            ohlc: DataFrame with OHLC data
            score_func: Scoring function ('profit_factor', 'sharpe', etc.)

        Returns:
            OptimisationResult with best parameters and score
        """
        param_space = self.get_parameter_space()
        # Calculate price returns without look-ahead bias
        # diff() gives return from t-1 to t: log(close[t]) - log(close[t-1])
        price_returns = np.log(ohlc['close']).diff()

        best_score = float('-inf')
        best_params = {}
        all_results = []

        param_combinations = self._generate_param_combinations(param_space)

        for params in param_combinations:
            try:
                result = self.generate_signal(ohlc, **params)
                # Align signal: signal at bar t-1 predicts return from t-1 to t
                aligned_signal = result.signal.shift(1)
                aligned_result = StrategyResult(signal=aligned_signal, metadata=result.metadata)
                score = self._calculate_score(aligned_result, price_returns, score_func)

                all_results.append((params.copy(), score))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                import traceback
                print(f"Error optimising params {params}: {e}")
                traceback.print_exc()
                continue

        return OptimisationResult(best_params, best_score, all_results)

    def _generate_param_combinations(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        if not param_space:
            return [{}]

        import itertools
        keys = list(param_space.keys())
        values = list(param_space.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _calculate_score(self, result: StrategyResult, price_returns: pd.Series, score_func: str) -> float:
        """Calculate score for optimisation."""
        if score_func == 'profit_factor':
            return result.get_profit_factor(price_returns)
        elif score_func == 'sharpe':
            strategy_returns = result.get_returns(price_returns)
            return strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        else:
            raise ValueError(f"Unknown score function: {score_func}")


class WalkForwardStrategy(ABC):
    """Base class for strategies that support walk-forward analysis."""

    @abstractmethod
    def walk_forward_signal(self, ohlc: pd.DataFrame, train_lookback: int = 24*365*4,
                           train_step: int = 24*30) -> pd.Series:
        """
        Generate walk-forward signals.

        Args:
            ohlc: DataFrame with OHLC data
            train_lookback: Number of periods for training window
            train_step: Number of periods between retraining

        Returns:
            Series with walk-forward signals
        """
        pass


class MCPTTester:
    """Monte Carlo Permutation Test framework."""

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy

    def _run_permutation_test(self, ohlc: pd.DataFrame, n_permutations: int, signal_func, signal_args, perm_kwargs=None, **extra_results) -> Dict[str, Any]:
        """
        Common MCPT logic for both in-sample and walk-forward testing.

        Args:
            ohlc: DataFrame with OHLC data
            n_permutations: Number of permutations to test
            signal_func: Function that takes ohlc and returns strategy returns
            perm_kwargs: Additional kwargs for get_permutation
            **extra_results: Additional results to include in return dict

        Returns:
            Dictionary with test results
        """

        # For row permutation, pre-calculate returns to avoid diff() on shuffled data
        # This prevents artificial returns from non-consecutive bars
        # For bar/block/ar1 modes, returns must be calculated from permuted prices
        # diff() gives return from t-1 to t: log(close[t]) - log(close[t-1])
        ohlc_with_returns = ohlc.copy()

        # Only precalculate returns for row permutation mode
        # For bar/block modes, permutation changes prices so returns must be recalculated
        perm_mode = perm_kwargs.get('permutation_mode', 'bar') if perm_kwargs else 'bar'
        if perm_mode == 'row':
            ohlc_with_returns['_precalc_returns'] = np.log(ohlc['close']).diff()

        real_returns = signal_func((ohlc_with_returns,) + signal_args)

        # Calculate cumulative return (primary test statistic for MCPT)
        real_cum_return = real_returns.sum()

        # Calculate profit factor for reporting (mark-to-market based, not trade-based)
        wins = real_returns[real_returns > 0].sum()
        losses = real_returns[real_returns < 0].abs().sum()
        real_pf = wins / losses if losses > 0 else np.inf if wins > 0 else 1.0

        real_cum_returns = real_returns.cumsum()

        max_workers = max(1, os.cpu_count() // 2)
        perm_better_count = 1
        permuted_cum_rets = []
        permuted_pfs = []
        permuted_cum_returns = []

        os.environ['TQDM_DISABLE'] = '1'

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            perm_args = [
                (ohlc_with_returns, signal_func, signal_args, perm_kwargs, (1000 + i))
                for i in range(1, n_permutations)
            ]
            futures = [executor.submit(_run_single_permutation, args) for args in perm_args]

            # Use tqdm only in main process for clean progress tracking
            disable_tqdm = mp.current_process().name != 'MainProcess'
            with tqdm(total=len(futures), desc="Running permutations", disable=disable_tqdm) as pbar:
                for future in as_completed(futures):
                    perm_cum_ret, perm_pf, perm_cum_ret_series = future.result()
                    # Compare using cumulative return (primary test statistic)
                    if perm_cum_ret >= real_cum_return:
                        perm_better_count += 1
                    permuted_cum_rets.append(perm_cum_ret)
                    permuted_pfs.append(perm_pf)
                    permuted_cum_returns.append(perm_cum_ret_series)
                    pbar.update(1)

        # Re-enable tqdm after multiprocessing
        if 'TQDM_DISABLE' in os.environ:
            del os.environ['TQDM_DISABLE']

        p_value = perm_better_count / n_permutations

        result = {
            'real_cumulative_return': real_cum_return,
            'permuted_cumulative_returns': permuted_cum_rets,
            'real_profit_factor': real_pf,
            'permuted_profit_factors': permuted_pfs,
            'real_cum_returns': real_cum_returns,
            'permuted_cum_returns': permuted_cum_returns,
            'p_value': p_value,
            'n_permutations': n_permutations,
        }
        result.update(extra_results)
        return result

    def run_insample_test(self, ohlc: pd.DataFrame, n_permutations: int = 1000,
                         perm_kwargs: dict = None, **strategy_params) -> Dict[str, Any]:
        """
        Run in-sample MCPT.

        Args:
            ohlc: DataFrame with OHLC data
            n_permutations: Number of permutations to test
            perm_kwargs: Additional kwargs for permutation (e.g., {'permutation_mode': 'row'})
            **strategy_params: Parameters for the strategy

        Returns:
            Dictionary with test results
        """
        if perm_kwargs is None:
            perm_kwargs = {}
        return self._run_permutation_test(
            ohlc, n_permutations, _insample_signal_func, (self.strategy, strategy_params),
            perm_kwargs=perm_kwargs,
            strategy_params=strategy_params
        )

    def run_walkforward_test(self, ohlc: pd.DataFrame, n_permutations: int = 200,
                           train_lookback: int = 24*365*4, train_step: int = 24*30,
                           perm_kwargs: dict = None) -> Dict[str, Any]:
        """
        Run walk-forward MCPT.

        Args:
            ohlc: DataFrame with OHLC data
            n_permutations: Number of permutations to test
            train_lookback: Training window size
            train_step: Retraining frequency
            perm_kwargs: Additional kwargs for permutation (e.g., {'permutation_mode': 'row'})

        Returns:
            Dictionary with test results
        """
        if not isinstance(self.strategy, WalkForwardStrategy):
            raise ValueError("Strategy must inherit from WalkForwardStrategy for walk-forward testing")

        if perm_kwargs is None:
            perm_kwargs = {}

        # Permute from start_index=0, i.e. the first training window too. Each
        # permuted series then goes through the SAME retrain-and-trade procedure as
        # the real one, so the null is "no temporal structure anywhere" and the
        # optimiser's own selection effect is part of both distributions.
        merged_perm_kwargs = {'start_index': 0, **perm_kwargs}

        return self._run_permutation_test(
            ohlc, n_permutations, _walkforward_signal_func, (self.strategy, train_lookback, train_step),
            perm_kwargs=merged_perm_kwargs,
            train_lookback=train_lookback,
            train_step=train_step
        )