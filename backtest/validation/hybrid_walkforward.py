"""
Hybrid walk-forward with holdout validation.

Phase 1 (development set): walk-forward optimisation. The strategy is re-optimised
every ``train_step`` bars on the trailing ``train_lookback`` bars, and only trades
the bars AFTER each training window (out-of-sample). The first training window is
never traded: a signal there would be generated with parameters fitted on the very
same bars, and its PnL would be in-sample performance dressed up as walk-forward.

Phase 2 (holdout set): the parameters selected most often in Phase 1 are frozen and
applied, bar by bar, to data the optimiser never saw.
"""
import argparse
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    def tqdm(iterable, *args, **kwargs):
        return iterable

from backtest.engine.config import get_strategy_by_name, list_strategies
from backtest.engine.trade_metrics import analyse_model_performance, get_model_trades
from backtest.validation.strategy_base import StrategyResult


def _filter_dates(ohlc: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=ohlc.index)
    if start_date is not None:
        mask &= (ohlc.index >= pd.to_datetime(start_date))
    if end_date is not None:
        mask &= (ohlc.index < pd.to_datetime(end_date))
    return ohlc[mask]


def _basis_params(train_data: pd.DataFrame) -> dict:
    """For spot/futures (basis) strategies: thresholds come from the training data
    ONLY, passed explicitly so the signal cannot calibrate on the bars it trades."""
    if 'spot' in train_data.columns and 'futures' in train_data.columns:
        return {'threshold_data': (train_data['futures'] - train_data['spot']) / train_data['spot']}
    return {}


def _signal_bar_by_bar(strategy, data: pd.DataFrame, start: int, end: int, params: dict) -> np.ndarray:
    """Signal for bars [start, end), each computed from data up to and including that
    bar only — no rolling computation can see a later bar."""
    out = np.zeros(end - start)
    for i in range(start, end):
        out[i - start] = strategy.generate_signal(data.iloc[:i + 1], **params).signal.iloc[-1]
    return out


def _cum_trade_pnl(result: StrategyResult, data: pd.DataFrame) -> pd.Series:
    """Cumulative trade PnL, booked at each trade's exit time."""
    trades = get_model_trades(result, data, position_threshold=0.0)
    cum = pd.Series(0.0, index=data.index)
    total = 0.0
    for _, trade in trades.iterrows():
        total += trade['pnl']
        cum.loc[trade['exit_time']:] = total
    return cum


def main(ohlc: pd.DataFrame,
         strategy_name: str,
         start_date: Optional[str] = None,
         end_date: Optional[str] = None,
         holdout_months: int = 4,
         plot: bool = True):
    """
    Run hybrid walk-forward with holdout validation.

    Args:
        ohlc: DataFrame with OHLC data (DatetimeIndex)
        strategy_name: Name of strategy to test (from the registry)
        start_date: Start date for data filtering (YYYY-MM-DD format)
        end_date: End date for data filtering (YYYY-MM-DD format)
        holdout_months: Number of months held back for the final validation
        plot: Draw the equity curves (requires matplotlib)
    """
    try:
        strategy = get_strategy_by_name(strategy_name)
        ohlc = _filter_dates(ohlc, start_date, end_date)
        print(f"Loaded data: {len(ohlc)} rows from {ohlc.index[0]} to {ohlc.index[-1]}")

        holdout_start = ohlc.index[-1] - pd.DateOffset(months=holdout_months)
        development_data = ohlc[ohlc.index < holdout_start]
        holdout_data = ohlc[ohlc.index >= holdout_start]

        print(f"\n{'='*60}\nDATA SPLIT\n{'='*60}")
        print(f"Development Set: {development_data.index[0]} to {development_data.index[-1]} ({len(development_data)} rows)")
        print(f"Holdout Set:     {holdout_data.index[0]} to {holdout_data.index[-1]} ({len(holdout_data)} rows)")
        print(f"Holdout is {len(holdout_data)/len(ohlc)*100:.1f}% of total data")

        # ------------------------------------------------------------------
        # Phase 1: walk-forward on the development set
        # ------------------------------------------------------------------
        print(f"\n{'='*60}\nPHASE 1: Walk-forward on the development set\n{'='*60}")
        n = len(development_data)
        if n < 2000:
            # Few rows -> probably daily bars: ~1 year training, ~quarter steps.
            train_lookback = min(n // 2, 252)
            train_step = max(train_lookback // 4, 60)
        else:
            # Intraday bars: larger windows.
            train_lookback = min(90000, max(60000, n // 4))
            train_step = min(30000, train_lookback // 3)
        print(f"Training Window: {train_lookback} periods | Retraining Step: {train_step} periods")

        if n <= train_lookback:
            print(f"ERROR: development set ({n} rows) is not larger than the training "
                  f"window ({train_lookback}). Reduce holdout_months or use more data.")
            return None

        wf_signal = pd.Series(0.0, index=development_data.index)
        selected_params_history = []
        for window_start in tqdm(range(train_lookback, n, train_step), desc="Walk-forward windows"):
            train_data = development_data.iloc[window_start - train_lookback:window_start]
            best = strategy.optimise(train_data).best_params
            selected_params_history.append(dict(best))

            window_end = min(window_start + train_step, n)
            params = {**best, **_basis_params(train_data)}
            wf_signal.iloc[window_start:window_end] = _signal_bar_by_bar(
                strategy, development_data, window_start, window_end, params)
            print(f"Window {len(selected_params_history)}: {best}")

        # Metrics only over the out-of-sample part (after the first training window).
        oos_data = development_data.iloc[train_lookback:]
        dev_result = StrategyResult(signal=wf_signal.iloc[train_lookback:],
                                    metadata={'strategy_name': strategy.name})
        dev_metrics = analyse_model_performance(dev_result, oos_data, position_threshold=0.0)
        dev_cum_returns = _cum_trade_pnl(dev_result, oos_data)

        # Most frequently selected parameters -> frozen for the holdout.
        param_counter = Counter(tuple(sorted(p.items())) for p in selected_params_history)
        print(f"\n{'='*60}\nPARAMETER SELECTION FREQUENCY\n{'='*60}")
        print(f"Total optimisation windows: {len(selected_params_history)}")
        for i, (param_tuple, count) in enumerate(param_counter.most_common(5), 1):
            pct = count / len(selected_params_history) * 100
            print(f"{i}. Selected {count}/{len(selected_params_history)} times ({pct:.1f}%): {dict(param_tuple)}")
        holdout_params = dict(param_counter.most_common(1)[0][0])

        # ------------------------------------------------------------------
        # Phase 2: frozen parameters on the holdout
        # ------------------------------------------------------------------
        print(f"\n{'='*60}\nPHASE 2: Holdout set (fixed parameters: {holdout_params})\n{'='*60}")
        full_data = pd.concat([development_data, holdout_data])
        params = {**holdout_params, **_basis_params(development_data)}
        holdout_signal = pd.Series(
            _signal_bar_by_bar(strategy, full_data, n, len(full_data), params),
            index=holdout_data.index)
        holdout_result = StrategyResult(signal=holdout_signal, metadata={'strategy_name': strategy.name})
        holdout_metrics = analyse_model_performance(holdout_result, holdout_data, position_threshold=0.0)
        holdout_cum_returns = _cum_trade_pnl(holdout_result, holdout_data)

        print(f"\n{'='*60}\nRESULTS\n{'='*60}")
        for label, m in (("DEVELOPMENT (walk-forward, out-of-sample only)", dev_metrics),
                         ("HOLDOUT (fixed parameters, never optimised on)", holdout_metrics)):
            print(f"\n{label}:")
            print(f"  Profit Factor: {m.get('Profit factor', 0):.4f}")
            print(f"  Total PnL: {m.get('Total pnl', m.get('total_pnl', 0)):.6f}")
            print(f"  Total Trades: {m.get('Total trades', m.get('total_trades', 0))}")
            print(f"  Win Rate: {m.get('Win rate', m.get('win_rate', 0))*100:.2f}%")

        dev_pf = dev_metrics.get('Profit factor', 0)
        holdout_pf = holdout_metrics.get('Profit factor', 0)
        if dev_pf > 0 and holdout_pf > 0 and np.isfinite(dev_pf):
            degradation = (dev_pf - holdout_pf) / dev_pf * 100
            print(f"\nPerformance Degradation: {degradation:.1f}%")
            if degradation > 50:
                print("WARNING: Severe performance degradation suggests overfitting")
            elif degradation > 25:
                print("CAUTION: Moderate performance degradation")
            elif degradation < 0:
                print("Holdout outperformed development")
            else:
                print("Acceptable performance degradation")

        if plot and plt is not None:
            plot_hybrid_results(strategy.name, dev_cum_returns, holdout_cum_returns, dev_pf, holdout_pf)

        return {
            'development_metrics': dev_metrics,
            'holdout_metrics': holdout_metrics,
            'holdout_params': holdout_params,
            'param_history': selected_params_history,
            'dev_cum_returns': dev_cum_returns,
            'holdout_cum_returns': holdout_cum_returns,
            'dev_signal': wf_signal,
            'train_lookback': train_lookback,
            'train_step': train_step,
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_hybrid_results(strategy_name: str, dev_cum_returns: pd.Series,
                        holdout_cum_returns: pd.Series, dev_pf: float, holdout_pf: float):
    import matplotlib.dates as mdates

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    panels = ((axes[0], dev_cum_returns, 'cyan', f'Development (OOS walk-forward) - PF: {dev_pf:.4f}'),
              (axes[1], holdout_cum_returns, 'red', f'Holdout Set (Fixed Params) - PF: {holdout_pf:.4f}'))
    for ax, series, color, title in panels:
        ax.plot(mdates.date2num(pd.to_datetime(series.index)), series.values, color=color, linewidth=2)
        ax.axhline(y=0, color='yellow', linestyle='--', linewidth=1, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Cumulative trade PnL', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{strategy_name} - Hybrid Walk-Forward with Holdout', fontsize=16)
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Hybrid Walk-Forward with Holdout Validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--strategy', '-s', help='Strategy name (registered in strategy registry)')
    parser.add_argument('--asset', '-a', default='SPY', help='Asset to load from the data lake')
    parser.add_argument('--list-strategies', action='store_true', help='List available strategies and exit')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--holdout-months', type=int, default=4,
                        help='Number of months to hold back for final validation')
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
    main(ohlc=data, strategy_name=args.strategy, start_date=args.start_date,
         end_date=args.end_date, holdout_months=args.holdout_months)
