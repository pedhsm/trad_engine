"""
Unified configuration and strategy registry for MCPT testing framework.
Consolidates MCPTConfig management and strategy registration.
"""
import json
import os
import importlib
from typing import Dict, Type, Any, Optional
from dataclasses import dataclass, asdict

from backtest.validation.strategy_base import TradingStrategy


# ---------------------------------------------------------------------------
# MCPT Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCPTConfig:
    """Configuration for MCPT testing."""
    data_file: str  # Required - no default, user must specify
    strategy_name: str  # Required - no default, user must specify
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    n_permutations: int = 1000

    # Walk-forward parameters
    train_lookback: int = 24 * 365 * 4
    train_step: int = 24 * 30

    # Output settings
    save_results: bool = True
    plot_results: bool = True
    output_dir: str = 'results'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPTConfig':
        """Create from dictionary."""
        return cls(**data)


class ConfigManager:
    """Manages MCPT configuration files."""

    @staticmethod
    def load_config(config_file: str) -> MCPTConfig:
        """
        Load configuration from JSON file.

        Args:
            config_file: Path to configuration file

        Returns:
            MCPTConfig instance
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found. Please create one or specify data_file and strategy_name.")

        try:
            with open(config_file, 'r') as f:
                data = json.load(f)

            # Validate required fields
            if 'data_file' not in data:
                raise ValueError("Config file must specify 'data_file'")
            if 'strategy_name' not in data:
                raise ValueError("Config file must specify 'strategy_name'")

            # Validate and create config
            config = MCPTConfig.from_dict(data)
            print(f"Loaded configuration from: {config_file}")
            return config

        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Error loading config file '{config_file}': {e}")

    @staticmethod
    def save_config(config: MCPTConfig, config_file: str):
        """
        Save configuration to JSON file.

        Args:
            config: MCPTConfig instance
            config_file: Path to save configuration
        """
        try:
            config_dir = os.path.dirname(config_file)
            if config_dir:  # Only create directory if path has a directory part
                os.makedirs(config_dir, exist_ok=True)

            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)

            print(f"Configuration saved to: {config_file}")

        except Exception as e:
            print(f"Error saving config file '{config_file}': {e}")

    @staticmethod
    def create_config_from_args(data_file: str, strategy_name: str, **kwargs) -> MCPTConfig:
        """
        Create configuration from command-line arguments.

        Args:
            data_file: Path to data file
            strategy_name: Name of strategy
            **kwargs: Additional configuration parameters

        Returns:
            MCPTConfig instance
        """
        return MCPTConfig(
            data_file=data_file,
            strategy_name=strategy_name,
            start_year=kwargs.get('start_year'),
            end_year=kwargs.get('end_year'),
            n_permutations=kwargs.get('n_permutations', 1000),
            train_lookback=kwargs.get('train_lookback', 24*365*4),
            train_step=kwargs.get('train_step', 24*30),
            save_results=kwargs.get('save_results', True),
            plot_results=kwargs.get('plot_results', True),
            output_dir=kwargs.get('output_dir', 'results')
        )

    @staticmethod
    def update_config_from_args(config: MCPTConfig, args) -> MCPTConfig:
        """
        Update configuration with command-line arguments.

        Args:
            config: Base configuration
            args: Parsed command-line arguments

        Returns:
            Updated configuration
        """

        if hasattr(args, 'data_file') and args.data_file:
            config.data_file = args.data_file
        if hasattr(args, 'strategy') and args.strategy:
            config.strategy_name = args.strategy
        if hasattr(args, 'start_year') and args.start_year is not None:
            config.start_year = args.start_year
        if hasattr(args, 'end_year') and args.end_year is not None:
            config.end_year = args.end_year
        if hasattr(args, 'permutations') and args.permutations is not None:
            config.n_permutations = args.permutations
        if hasattr(args, 'train_lookback') and args.train_lookback is not None:
            config.train_lookback = args.train_lookback
        if hasattr(args, 'train_step') and args.train_step is not None:
            config.train_step = args.train_step

        return config


def print_config(config: MCPTConfig):
    print("MCPT Config")
    print(f"Data File: {config.data_file}")
    print(f"Strategy: {config.strategy_name}")
    print(f"Time Period: {config.start_year}-{config.end_year}")
    print(f"Permutations: {config.n_permutations}")
    print(f"Training Window: {config.train_lookback} periods")
    print(f"Retraining Step: {config.train_step} periods")
    print(f"Save Results: {config.save_results}")
    print(f"Plot Results: {config.plot_results}")
    print(f"Output Directory: {config.output_dir}")
    print("=" * 27)


# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------

class StrategyRegistry:
    """Registry for managing strategy name-to-implementation mappings."""

    def __init__(self, registry_file: Optional[str] = None):
        # A JSON file of extra strategies (written by register_strategy). Optional:
        # without it the built-in examples below are available.
        registry_file = registry_file or os.environ.get('TRAD_ENGINE_REGISTRY', 'strategies.reg.json')
        self.registry_file = registry_file
        self.strategies: Dict[str, Dict[str, str]] = {}
        self.loaded_strategies: Dict[str, Type[TradingStrategy]] = {}
        self.load_registry()

    def load_registry(self):
        """Load strategy mappings from registry file."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    self.strategies = json.load(f)
                # Suppress loading message in subprocesses
                if os.getenv('MCPT_SUBPROCESS') != '1':
                    print(f"Loaded {len(self.strategies)} strategies from {self.registry_file}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading registry: {e}")
                self.create_default_registry()
        else:
            self.create_default_registry()

    def save_registry(self):
        """Save current strategy mappings to registry file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.strategies, f, indent=2)
            print(f"Registry saved to {self.registry_file}")
        except Exception as e:
            print(f"Error saving registry: {e}")

    def create_default_registry(self):
        """Create default strategy registry."""
        self.strategies = {
            "example: donchian": {
                "module": "backtest.strategies._legacy.donchian_strategy",
                "class": "DonchianStrategy",
                "description": "Donchian breakout (example)"
            },
            "example: bollinger": {
                "module": "backtest.strategies._legacy.bollinger_bands",
                "class": "BollingerBands",
                "description": "Bollinger bands mean reversion (example)"
            },
            "example: basis": {
                "module": "backtest.strategies._legacy.ddm_v1",
                "class": "DDMStrategy",
                "description": "Spot/futures basis percentile model (example; needs spot/futures columns)"
            },
        }
        # In-memory only: do not write a registry file at import time.

    def register_strategy(self, name: str, module_path: str, class_name: str, description: str = ""):
        """
        Register a new strategy.

        Args:
            name: Strategy identifier name
            module_path: Python import path (e.g., 'strategies.moving_average')
            class_name: Class name in the module (e.g., 'MovingAverageStrategy')
            description: Human-readable description
        """
        self.strategies[name] = {
            "module": module_path,
            "class": class_name,
            "description": description or f"{class_name} strategy"
        }
        self.save_registry()
        print(f"Registered strategy '{name}' -> {module_path}.{class_name}")

    def get_strategy(self, name: str) -> TradingStrategy:
        """
        Load and return strategy instance by name.

        Args:
            name: Strategy name from registry (case-insensitive)

        Returns:
            Strategy instance
        """
        # Convert to lowercase for case-insensitive lookup
        name_lower = name.lower()

        # Check loaded strategies (also case-insensitive)
        for loaded_name in self.loaded_strategies:
            if loaded_name.lower() == name_lower:
                return self.loaded_strategies[loaded_name]()

        # Find matching strategy name (case-insensitive)
        actual_name = None
        for strategy_name in self.strategies:
            if strategy_name.lower() == name_lower:
                actual_name = strategy_name
                break

        if actual_name is None:
            available = ', '.join(self.strategies.keys())
            raise ValueError(f"Strategy '{name}' not found. Available: {available}")

        strategy_info = self.strategies[actual_name]

        try:
            # Import the module
            module = importlib.import_module(strategy_info["module"])

            # Get the class
            strategy_class = getattr(module, strategy_info["class"])

            # Verify it's a valid strategy
            if not issubclass(strategy_class, TradingStrategy):
                raise ValueError(f"Class {strategy_info['class']} is not a TradingStrategy")

            # Cache the class using actual name
            self.loaded_strategies[actual_name] = strategy_class

            # Return instance
            return strategy_class()

        except ImportError as e:
            raise ImportError(f"Cannot import {strategy_info['module']}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Class {strategy_info['class']} not found in {strategy_info['module']}: {e}")

    def list_strategies(self) -> Dict[str, str]:
        """List all registered strategies with descriptions."""
        return {name.upper(): info["description"] for name, info in self.strategies.items()}

    def remove_strategy(self, name: str):
        """Remove a strategy from the registry."""
        if name in self.strategies:
            del self.strategies[name]
            if name in self.loaded_strategies:
                del self.loaded_strategies[name]
            self.save_registry()
            print(f"Removed strategy '{name}'")
        else:
            print(f"Strategy '{name}' not found in registry")


# Global registry instance
registry = StrategyRegistry()


def get_strategy_by_name(name: str) -> TradingStrategy:
    """Convenience function to get strategy by name."""
    return registry.get_strategy(name)


def register_strategy(name: str, module_path: str, class_name: str, description: str = ""):
    """Convenience function to register a strategy."""
    registry.register_strategy(name, module_path, class_name, description)


def list_strategies():
    """List all available strategies."""
    strategies = registry.list_strategies()
    print("Available Strategies:")
    for name, desc in strategies.items():
        print(f"  {name}: {desc}")
    return strategies


if __name__ == '__main__':
    """Command-line interface for managing strategy registry."""
    import argparse

    parser = argparse.ArgumentParser(description='Manage strategy registry and configuration')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List strategies
    list_parser = subparsers.add_parser('list', help='List all strategies')

    # Register strategy
    register_parser = subparsers.add_parser('register', help='Register new strategy')
    register_parser.add_argument('name', help='Strategy name')
    register_parser.add_argument('module', help='Module path (e.g., strategies.moving_average)')
    register_parser.add_argument('class_name', help='Class name (e.g., MovingAverageStrategy)')
    register_parser.add_argument('--description', '-d', help='Strategy description')

    # Remove strategy
    remove_parser = subparsers.add_parser('remove', help='Remove strategy')
    remove_parser.add_argument('name', help='Strategy name to remove')

    # Test strategy
    test_parser = subparsers.add_parser('test', help='Test loading a strategy')
    test_parser.add_argument('name', help='Strategy name to test')

    args = parser.parse_args()

    if args.command == 'list':
        list_strategies()
    elif args.command == 'register':
        register_strategy(args.name, args.module, args.class_name, args.description or "")
    elif args.command == 'remove':
        registry.remove_strategy(args.name)
    elif args.command == 'test':
        try:
            strategy = get_strategy_by_name(args.name)
            print(f"Successfully loaded: {strategy.name}")
        except Exception as e:
            print(f"Error loading strategy: {e}")
    else:
        parser.print_help()
