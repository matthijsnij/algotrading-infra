"""
================================================================================
CONFIG LOADER
================================================================================
Load YAML config files and validate with pydantic models.

Functions:
    load_backtest_config(path: str) -> BacktestConfig
    load_live_config(path: str) -> LiveConfig
================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar
import yaml
from pydantic import BaseModel, ValidationError

from lighthouse.config.models import BacktestConfig, LiveConfig

ModelT = TypeVar("ModelT", bound=BaseModel)


def _load(config_path: str | Path, model: type[ModelT]) -> ModelT:
    """
    Read a YAML file and validate it against the given pydantic model.

    Args:
        config_path: Path to the YAML config file
        model: pydantic model class to validate against

    Returns:
        A validated instance of `model`

    Raises:
        FileNotFoundError: Config file not found
        ValueError: YAML parsing or pydantic validation failed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML in '{config_path}': {e}") from e

    if data is None:
        raise ValueError(f"Config file is empty: {config_path}")

    try:
        return model(**data)
    except ValidationError as e:
        raise ValueError(f"Config validation failed for '{config_path}':\n{e}") from e


def load_backtest_config(config_path: str | Path) -> BacktestConfig:
    """Load and validate a backtest config YAML file (see examples/backtest.yaml)."""
    return _load(config_path, BacktestConfig)


def load_live_config(config_path: str | Path) -> LiveConfig:
    """Load and validate a live/testnet config YAML file (see examples/live.yaml, examples/testnet.yaml)."""
    return _load(config_path, LiveConfig)
