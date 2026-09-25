"""
Strategy template.
Shows how a strategy plugs into the engine's core_math primitives.
"""
import pandas as pd
from typing import Dict, Any, List

from backtest.validation.strategy_base import TradingStrategy, StrategyResult
# Example import of microstructure primitives:
# from core_math.micro_math import compute_vpin, compute_tib

class TemplateSpyMicroStrategy(TradingStrategy):
    """
    Strategy skeleton consuming microstructure metrics
    from the core_math package.
    """

    def __init__(self):
        super().__init__("TemplateSPYMicro")

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Defines the parameter search space.
        """
        return {
            'vpin_threshold': [0.7, 0.8, 0.9],
            'tib_window': [10, 20, 50]
        }

    def generate_signal(self, ohlc: pd.DataFrame, **params) -> StrategyResult:
        """
        Generates the signals using the core's math modules.
        """
        # Extract parameters
        vpin_threshold = params.get('vpin_threshold', 0.8)
        tib_window = params.get('tib_window', 20)

        # -------------------------------------------------------------
        # Example of where the core math would be called:
        #
        # vpin_series = compute_vpin(ohlc, ...)
        # tib_series = compute_tib(ohlc, window=tib_window)
        #
        # Strategy logic...
        # -------------------------------------------------------------

        # Initialize neutral signal
        signal = pd.Series(0.0, index=ohlc.index)

        return StrategyResult(
            signal=signal,
            metadata={
                'strategy_name': self.name,
                'params_used': params
            }
        )
