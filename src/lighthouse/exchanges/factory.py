"""
================================================================================
EXCHANGE FACTORY MODULE 
================================================================================

This module contains the exchange registry and functionality to instantiate exchange objects for live trading.

Note that the registry only includes actual exchanges, i.e. not the BaseExchange and BacktestExchange classes.
- BaseExchange is not meant to be instantiated.
- BacktestExchange is only used for backtesting.

Functions:
    create_exchange(): Factory function to create an exchange object by name.
================================================================================
"""

from lighthouse.exchanges.live.phemex import PhemexExchange
from lighthouse.exchanges.live.alpaca import AlpacaExchange
from lighthouse.exchanges.base import BaseExchange

# ── Registry ──────────────────────────────────────────────────────────────────
_EXCHANGE_REGISTRY = {
    "phemex": PhemexExchange,
    "alpaca": AlpacaExchange,
}

# ── Factory ──────────────────────────────────────────────────────────────────
def create_exchange(
    exchange_name: str,
    *,
    credentials: dict[str, str],
    market_type: str,
    testnet: bool = False,
) -> BaseExchange:
    '''
    Creates a BaseExchange object corresponding to an input exchange.

    Args:
        exchange_name: the name of the exchange
        credentials: dict with "api_key" and "api_secret" (see config.credentials.resolve_credentials)
        market_type: supported values are exchange-specific, see the target exchange class's SUPPORTED_MARKET_TYPES
        testnet: if True, connect to the exchange's sandbox/testnet environment
    
    Returns:
        A BaseExchange instance of the corresponding exchange class

    Raises:
        ValueError: if the exchange is not in the registry, does not support testnet mode,
            or does not support the given market_type
    '''
    if exchange_name.lower() not in _EXCHANGE_REGISTRY:
        raise ValueError(f"Exchange '{exchange_name}' not supported.")
    
    exchange_cls = _EXCHANGE_REGISTRY[exchange_name.lower()]
    
    if testnet and not getattr(exchange_cls, "supports_testnet", False):
        supported = [
            name for name, cls in _EXCHANGE_REGISTRY.items()
            if getattr(cls, "supports_testnet", False)
        ]
        raise ValueError(
            f"Exchange '{exchange_name}' does not support testnet/sandbox mode. "
            f"Exchanges that do: {supported}. Set testnet: false or pick one of those."
        )
    return exchange_cls(
        credentials=credentials, market_type=market_type, testnet=testnet,
    )