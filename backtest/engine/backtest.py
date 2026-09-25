"""
Unified test runner for strategy analysis.
Supports both interactive mode and command-line arguments.
"""
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

from backtest.engine.config import get_strategy_by_name, list_strategies
from backtest.engine.trade_metrics import analyse_model_performance, print_metrics
from backtest.validation.mcpt_runner import MCPTRunner, run_insample, run_walkforward
from backtest.validation import hybrid_walkforward, pure_walkforward
import pandas as pd


class DuckDBDataLoader:
    """Dynamic S3/Parquet data loader via DuckDB."""
    
    @staticmethod
    def load_data(asset: str, interval: str = '1min', start_date: str = None, end_date: str = None) -> pd.DataFrame:
        try:
            import duckdb  # optional dependency: only this loader needs it
        except ImportError as e:
            raise ImportError("DuckDBDataLoader needs duckdb: pip install duckdb") from e
        try:
            con = duckdb.connect()
            
            # Install and load httpfs
            con.execute("INSTALL httpfs; LOAD httpfs;")

            # Load S3 credentials
            s3_endpoint = os.environ.get('S3_ENDPOINT_URL', 'localhost:9000').replace('http://', '').replace('https://', '')
            # No credential defaults in code: set them in the environment.
            s3_access = os.environ.get('S3_ACCESS_KEY')
            s3_secret = os.environ.get('S3_SECRET_KEY')
            if not s3_access or not s3_secret:
                raise ValueError("Set S3_ACCESS_KEY and S3_SECRET_KEY in the environment.")
            s3_bucket = os.environ.get('S3_BUCKET', 'my-market-data')
            
            con.execute(f"SET s3_endpoint='{s3_endpoint}';")
            con.execute(f"SET s3_access_key_id='{s3_access}';")
            con.execute(f"SET s3_secret_access_key='{s3_secret}';")
            if 'localhost' in s3_endpoint or '127.0.0.1' in s3_endpoint:
                con.execute("SET s3_use_ssl=false;")
            con.execute("SET s3_url_style='path';")
            
            # Layout is configurable — set S3_PREFIX to match your lake's partitioning.
            s3_prefix = os.environ.get('S3_PREFIX', 'bars/l1').strip('/')
            s3_path = f"s3://{s3_bucket}/{s3_prefix}/{interval}/{asset}/**/*.parquet"
            
            # Build query
            query = f"SELECT ts, open, high, low, close, volume FROM read_parquet('{s3_path}', hive_partitioning=1)"
            
            conditions = []
            if start_date:
                conditions.append(f"ts >= '{start_date}'")
            if end_date:
                conditions.append(f"ts <= '{end_date}'")
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY ts"
            
            print(f"Running S3 query: {query}")
            df = con.execute(query).df()
            
            if df.empty:
                raise ValueError(f"No data found for {asset} in S3.")
            
            required_cols = ['open', 'high', 'low', 'close']
            df_cols_lower = [col.lower() for col in df.columns]
            missing_cols = [col for col in required_cols if col not in df_cols_lower]
            if missing_cols:
                raise ValueError(f"Data must contain columns: {required_cols}. Missing: {missing_cols}")
            
            col_mapping = {col.lower(): col for col in df.columns}
            rename_dict = {col_mapping[req_col]: req_col for req_col in required_cols if req_col in df_cols_lower}
            df = df.rename(columns=rename_dict)

            time_cols = [col for col in df.columns if col.lower() in ['date', 'timestamp', 'time', 'datetime', 'ts']]
            if time_cols:
                df.set_index(time_cols[0], inplace=True)
                df.index.name = 'Date'
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, format='ISO8601', errors='coerce')
            
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            df.index = df.index.astype("datetime64[s]")
                
            return df
        except Exception as e:
            raise ValueError(f"Error loading data from DuckDB S3: {e}")
        finally:
            if 'con' in locals():
                con.close()


# ---------------------------------------------------------------------------
# Core analysis function (consolidated from runner_core.py)
# ---------------------------------------------------------------------------

def run_strategy_analysis(ohlc: pd.DataFrame, strategy_name: str,
                         test_type: str = 'metrics',
                         position_threshold: float = 0.0,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         n_permutations: int = 1000,
                         permutation_mode: str = 'bar',
                         block_length: Optional[int] = None):
    """
    Unified runner for strategy testing and analysis.

    Args:
        ohlc: DataFrame with OHLC data
        strategy_name: Name of strategy from registry
        test_type: Type of test ('metrics', 'insample', 'walkforward', 'both')
        position_threshold: Threshold for determining position changes
        start_date: Start date for data filtering (YYYY-MM-DD format)
        end_date: End date for data filtering (YYYY-MM-DD format)
        n_permutations: Number of permutations for MCPT tests
        permutation_mode: Permutation mode ('bar', 'row', 'ar1', or 'block') for MCPT tests
        block_length: Block length for block bootstrap (only used if permutation_mode='block')
    """
    try:
        # Get strategy from registry
        strategy = get_strategy_by_name(strategy_name)

        # Load data using MCPTRunner
        runner = MCPTRunner(strategy)
        
        # We manually filter data since we removed load_data from MCPTRunner
        if start_date is not None or end_date is not None:
            mask = pd.Series(True, index=ohlc.index)
            if start_date is not None:
                start_dt = pd.to_datetime(start_date)
                mask &= (ohlc.index >= start_dt)
            if end_date is not None:
                end_dt = pd.to_datetime(end_date)
                mask &= (ohlc.index < end_dt)
            runner.data = ohlc[mask]
        else:
            runner.data = ohlc

        print(f"Strategy: {strategy.name}")
        print(f"Data: {len(runner.data)} rows from {runner.data.index[0]} to {runner.data.index[-1]}")

        if test_type == 'metrics':
            print('Metrics:')
            strategy_result = strategy.generate_signal(runner.data)
            metrics = analyse_model_performance(strategy_result, runner.data, position_threshold)
            print_metrics(metrics)
            return metrics

        elif test_type == 'insample':
            print(f"Running In-Sample MCPT ({n_permutations} permutations)")
            return run_insample(
                ohlc=runner.data,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                n_permutations=n_permutations,
                permutation_mode=permutation_mode,
                block_length=block_length
            )

        elif test_type == 'walkforward':
            print(f"Running Walk-Forward MCPT ({n_permutations} permutations)")

            # Auto-calculate reasonable defaults based on data size
            data_size = len(runner.data)
            train_lookback = min(data_size // 3, 20000)  # Use 1/3 of data or 20k, whichever is smaller
            # Scale train_step based on data size: 1/6 of lookback, with min based on dataset size
            # For small datasets (<5000), use smaller steps; for large datasets, cap at 1000
            min_step = min(max(data_size // 20, 10), 1000)
            train_step = max(train_lookback // 6, min_step)

            print(f"Auto-configured: train_lookback={train_lookback}, train_step={train_step}")

            return run_walkforward(
                ohlc=runner.data,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                n_permutations=n_permutations,
                train_lookback=train_lookback,
                train_step=train_step,
                permutation_mode=permutation_mode,
                block_length=block_length
            )

        elif test_type == 'both':
            print(f"1. Running In-Sample MCPT ({n_permutations} permutations)")
            insample_results = run_insample(
                ohlc=runner.data,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                n_permutations=n_permutations,
                permutation_mode=permutation_mode,
                block_length=block_length
            )

            print(f"\n2. Running Walk-Forward MCPT ({n_permutations} permutations)")

            # Auto-calculate reasonable defaults based on data size
            data_size = len(runner.data)
            train_lookback = min(data_size // 3, 20000)
            min_step = min(max(data_size // 20, 10), 1000)
            train_step = max(train_lookback // 6, min_step)

            print(f"Auto-configured: train_lookback={train_lookback}, train_step={train_step}")

            walkforward_results = run_walkforward(
                ohlc=runner.data,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                n_permutations=n_permutations,
                train_lookback=train_lookback,
                train_step=train_step,
                permutation_mode=permutation_mode,
                block_length=block_length
            )

            return {'insample': insample_results, 'walkforward': walkforward_results}
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    except Exception as e:
        import traceback
        print(f"Error running analysis: {e}")
        traceback.print_exc()
        return None


class InteractiveRunner:
    """Interactive CLI for running strategy tests."""

    MAIN_CATEGORIES = {
        '1': {
            'name': 'metrics',
            'display': 'Performance Metrics',
            'description': 'Calculate trading metrics only (no validation)',
        },
        '2': {
            'name': 'mcpt',
            'display': 'MCPT Tests',
            'description': 'Monte Carlo Permutation Tests for statistical validation',
        },
        '3': {
            'name': 'walkforward',
            'display': 'Walk-Forward Tests',
            'description': 'Rolling window walk-forward validation',
        }
    }

    MCPT_TESTS = {
        '1': {
            'name': 'insample',
            'display': 'In-Sample MCPT',
            'description': 'Test on entire dataset with permutations',
            'params': ['strategy', 'asset', 'n_permutations', 'permutation_mode', 'start_date', 'end_date']
        },
        '2': {
            'name': 'walkforward_mcpt',
            'display': 'Walk-Forward MCPT',
            'description': 'Rolling window with permutation testing',
            'params': ['strategy', 'asset', 'n_permutations', 'permutation_mode', 'start_date', 'end_date']
        },
        '3': {
            'name': 'both_mcpt',
            'display': 'Both MCPT Tests',
            'description': 'Run both in-sample and walk-forward MCPT',
            'params': ['strategy', 'asset', 'n_permutations', 'permutation_mode', 'start_date', 'end_date']
        }
    }

    WALKFORWARD_TESTS = {
        '1': {
            'name': 'simple_walkforward',
            'display': 'Simple Walk-Forward',
            'description': 'Pure rolling window validation (no permutations)',
            'params': ['strategy', 'asset', 'train_lookback', 'train_step', 'start_date', 'end_date']
        },
        '2': {
            'name': 'hybrid_walkforward',
            'display': 'Hybrid Walk-Forward',
            'description': 'Walk-forward optimisation + holdout validation',
            'params': ['strategy', 'asset', 'holdout_months', 'start_date', 'end_date']
        }
    }

    PARAM_DESCRIPTIONS = {
        'strategy': {
            'prompt': 'Enter strategy name',
            'required': True,
            'default': None,
            'help': 'Strategy registered in strats.reg.json'
        },
        'asset': {
            'prompt': 'Enter asset symbol (e.g. SPY, EURUSD)',
            'required': True,
            'default': 'SPY',
            'help': 'Asset symbol to pull from S3'
        },
        'n_permutations': {
            'prompt': 'Enter number of permutations',
            'required': False,
            'default': '1000',
            'help': 'Number of random permutations for MCPT'
        },
        'permutation_mode': {
            'prompt': 'Enter permutation mode (bar/row/ar1)',
            'required': False,
            'default': 'bar',
            'help': 'bar: conservative (default), row: preserves correlations, ar1: preserves autocorrelation (most conservative for mean reversion)'
        },
        'position_threshold': {
            'prompt': 'Enter position threshold',
            'required': False,
            'default': '0.0',
            'help': 'Threshold for determining position changes'
        },
        'train_lookback': {
            'prompt': 'Enter training window size (periods) [optional]',
            'required': False,
            'default': None,
            'help': 'Auto-configured if not provided'
        },
        'train_step': {
            'prompt': 'Enter retraining step (periods) [optional]',
            'required': False,
            'default': None,
            'help': 'Auto-configured if not provided'
        },
        'holdout_months': {
            'prompt': 'Enter holdout period (months)',
            'required': False,
            'default': '4',
            'help': 'Months to hold back for validation'
        },
        'start_date': {
            'prompt': 'Enter start date (YYYY-MM-DD) [optional]',
            'required': False,
            'default': None,
            'help': 'Filter data from this date onwards'
        },
        'end_date': {
            'prompt': 'Enter end date (YYYY-MM-DD) [optional]',
            'required': False,
            'default': None,
            'help': 'Filter data until this date'
        }
    }

    def __init__(self):
        self.params: Dict[str, Any] = {}

    def print_banner(self):
        """Print welcome banner."""
        print("\n" + "="*60)
        print("         STRATEGY TESTING")
        print("="*60)
        print("Select a test type to run strategy analysis\n")

    def display_main_menu(self) -> str:
        print("Test Categories:")

        for key, category_info in self.MAIN_CATEGORIES.items():
            print(f"{key}. {category_info['display']}")
            print(f"   {category_info['description']}\n")

        print("0. Exit")
        print("-" * 60)

        while True:
            choice = input("\nSelect category (0-3): ").strip()

            if choice == '0':
                print("Exiting...")
                sys.exit(0)

            if choice in self.MAIN_CATEGORIES:
                return self.MAIN_CATEGORIES[choice]['name']

            print("Invalid choice. Please enter 0-3.")

    def display_submenu(self, category: str) -> tuple:
        if category == 'metrics':
            # Metrics has no submenu
            return ('metrics', {
                'display': 'Performance Metrics',
                'params': ['strategy', 'position_threshold', 'start_date', 'end_date']
            })

        elif category == 'mcpt':
            print("MCPT Test Options:")

            for key, test_info in self.MCPT_TESTS.items():
                print(f"{key}. {test_info['display']}")
                print(f"   {test_info['description']}\n")

            print("0. Back")
            print("-" * 60)

            while True:
                choice = input("\nSelect MCPT test (0-3): ").strip()

                if choice == '0':
                    return None

                if choice in self.MCPT_TESTS:
                    test = self.MCPT_TESTS[choice]
                    return (test['name'], test)

                print("Invalid choice. Please enter 0-3.")

        elif category == 'walkforward':
            print("Walk-Forward Test Options:")

            for key, test_info in self.WALKFORWARD_TESTS.items():
                print(f"{key}. {test_info['display']}")
                print(f"   {test_info['description']}\n")

            print("0. Back")
            print("-" * 60)

            while True:
                choice = input("\nSelect walk-forward test (0-2): ").strip()

                if choice == '0':
                    return None

                if choice in self.WALKFORWARD_TESTS:
                    test = self.WALKFORWARD_TESTS[choice]
                    return (test['name'], test)

                print("Invalid choice. Please enter 0-2.")

    def list_available_strategies(self):
        print("Available Strategies:")
        print("-" * 60)

        try:
            # Import registry directly to avoid double-printing
            from backtest.engine.config import registry
            strategies = registry.list_strategies()
            if strategies:
                for name, description in strategies.items():
                    print(f"  * {name}: {description}")
            else:
                print("  No strategies registered")
        except Exception as e:
            print(f"  Error loading strategies: {e}")

        print("-" * 60 + "\n")

    def get_parameter(self, param_name: str) -> Optional[str]:
        """Get parameter value from user."""
        param_info = self.PARAM_DESCRIPTIONS[param_name]

        # Special handling for strategy parameter
        if param_name == 'strategy':
            show_strategies = input("Show available strategies? (y/n) [y]: ").strip().lower()
            if show_strategies != 'n':
                self.list_available_strategies()

        # Build prompt
        prompt = f"{param_info['prompt']}"
        if param_info['default']:
            prompt += f" [{param_info['default']}]"
        prompt += ": "

        # Get input
        value = input(prompt).strip()

        # Handle empty input
        if not value:
            if param_info['required']:
                print(f"Error: {param_name} is required!")
                return self.get_parameter(param_name)
            else:
                return param_info['default']

        return value

    def collect_parameters(self, test_name: str, test_info: dict):
        """Collect parameters for selected test type."""
        print(f"Configuring: {test_info.get('display', test_name)}")

        # Collect each required parameter
        for param_name in test_info['params']:
            value = self.get_parameter(param_name)

            if value is None and self.PARAM_DESCRIPTIONS[param_name]['required']:
                print(f"\nCancelled: Required parameter '{param_name}' not provided.")
                return False

            self.params[param_name] = value

        return True

    def confirm_and_run(self, test_name: str):
        print(f"Test Type: {test_name}")

        for param, value in self.params.items():
            if value is not None:
                print(f"{param}: {value}")

        confirm = input("Run test with these parameters? (y/n) [y]: ").strip().lower()

        if confirm == 'n':
            print("Test cancelled.")
            return None

        print("-" * 60 + "\n")
        print("Running Test...")

        # Route to appropriate test based on test_name
        
        # Load Data
        print(f"Loading data from DuckDB (S3) for {self.params.get('asset', 'SPY')}...")
        ohlc = DuckDBDataLoader.load_data(self.params.get('asset', 'SPY'), start_date=self.params.get('start_date'), end_date=self.params.get('end_date'))
        
        if test_name == 'metrics':
            position_threshold = float(self.params.get('position_threshold', 0.0)) if self.params.get('position_threshold') else 0.0
            results = run_strategy_analysis(
                ohlc=ohlc,
                strategy_name=self.params['strategy'],
                test_type='metrics',
                position_threshold=position_threshold,
                start_date=self.params.get('start_date'),
                end_date=self.params.get('end_date'),
                n_permutations=1000
            )

        elif test_name in ['insample', 'walkforward_mcpt', 'both_mcpt']:
            n_permutations = int(self.params.get('n_permutations', 1000)) if self.params.get('n_permutations') else 1000
            permutation_mode = self.params.get('permutation_mode', 'bar')
            # Map test names to runner_core names
            test_type_map = {
                'insample': 'insample',
                'walkforward_mcpt': 'walkforward',
                'both_mcpt': 'both'
            }
            results = run_strategy_analysis(
                ohlc=ohlc,
                strategy_name=self.params['strategy'],
                test_type=test_type_map[test_name],
                position_threshold=0.0,
                start_date=self.params.get('start_date'),
                end_date=self.params.get('end_date'),
                n_permutations=n_permutations,
                permutation_mode=permutation_mode
            )

        elif test_name == 'simple_walkforward':
            train_lookback = int(self.params['train_lookback']) if self.params.get('train_lookback') else None
            train_step = int(self.params['train_step']) if self.params.get('train_step') else None
            results = pure_walkforward.main(
                ohlc=ohlc,
                strategy_name=self.params['strategy'],
                start_date=self.params.get('start_date'),
                end_date=self.params.get('end_date'),
                train_lookback=train_lookback,
                train_step=train_step
            )

        elif test_name == 'hybrid_walkforward':
            holdout_months = int(self.params.get('holdout_months', 4))
            results = hybrid_walkforward.main(
                ohlc=ohlc,
                strategy_name=self.params['strategy'],
                start_date=self.params.get('start_date'),
                end_date=self.params.get('end_date'),
                holdout_months=holdout_months
            )

        else:
            print(f"Unknown test type: {test_name}")
            return None

        return results

    def run(self):
        self.print_banner()

        # Level 1: Select category
        while True:
            category = self.display_main_menu()

            # Level 2: Select specific test (or straight to params for metrics)
            test_selection = self.display_submenu(category)

            if test_selection is None:
                # User chose "Back", return to main menu
                print()
                continue

            test_name, test_info = test_selection

            # Collect parameters
            if not self.collect_parameters(test_name, test_info):
                sys.exit(1)

            # Confirm and run
            results = self.confirm_and_run(test_name)

            if results is None:
                sys.exit(1)

            print(f"\n{'='*60}")
            print("Test Completed Successfully")
            print(f"{'='*60}\n")

            # Exit after successful run
            break


def run_cli_mode():
    """Run in traditional command-line mode."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Strategy Test Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('test_type', nargs='?',
                       choices=['metrics', 'insample', 'walkforward', 'both'],
                       help='Type of analysis to run')



    parser.add_argument('--strategy', '-s',
                       help='Strategy name from registry')

    parser.add_argument('--asset', '-a', type=str, default='SPY',
                       help='Asset symbol to pull from S3 (e.g., SPY, EURUSD)')

    parser.add_argument('--list-strategies', action='store_true',
                       help='List available strategies and exit')

    parser.add_argument('--permutations', '-p', type=int, default=1000,
                       help='Number of permutations for MCPT tests')

    parser.add_argument('--start-date', type=str,
                       help='Start date for data filtering (YYYY-MM-DD format)')

    parser.add_argument('--end-date', type=str,
                       help='End date for data filtering (YYYY-MM-DD format)')

    parser.add_argument('--position-threshold', type=float, default=0.0,
                       help='Threshold for determining position changes')

    parser.add_argument('--permutation-mode', type=str, default='bar',
                       choices=['bar', 'row', 'ar1', 'block'],
                       help='MCPT permutation mode: bar (conservative, default), row (preserves correlations), ar1 (preserves autocorrelation), or block (moving-block bootstrap, gold standard for autocorrelated data)')

    parser.add_argument('--block-length', type=int, default=None,
                       help='Block length for block bootstrap (default: 10 bars for daily data, optimal by n^(1/3) rule). Range: 5-20 bars recommended.')

    args = parser.parse_args()

    # List strategies
    if args.list_strategies:
        list_strategies()
        return

    # Interactive mode (when no required args provided)
    if not args.test_type and not args.strategy:
        runner = InteractiveRunner()
        runner.run()
        return

    # Validate required arguments for CLI mode
    if not args.test_type:
        parser.error("test_type is required in CLI mode")
    if not args.strategy:
        parser.error("--strategy is required in CLI mode")

    # Run CLI mode
    print(f"Loading data from DuckDB (S3) for {args.asset}...")
    ohlc = DuckDBDataLoader.load_data(args.asset, start_date=args.start_date, end_date=args.end_date)
    
    results = run_strategy_analysis(
        ohlc=ohlc,
        strategy_name=args.strategy,
        test_type=args.test_type,
        position_threshold=args.position_threshold,
        start_date=args.start_date,
        end_date=args.end_date,
        n_permutations=args.permutations,
        permutation_mode=args.permutation_mode,
        block_length=args.block_length
    )

    if results is None:
        sys.exit(1)


def main():
    """Main entry point."""
    run_cli_mode()


if __name__ == "__main__":
    main()