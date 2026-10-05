"""
================================================================================
CREDENTIALS RESOLVER
================================================================================
Resolve API keys and secrets from environment variables or keyring.
No secrets hardcoded in YAML or code.

Functions:
    resolve_credentials(exchange: str, testnet: bool = False) -> dict[str, str]
    check_credentials_not_shared(exchange: str) -> None
    resolve_telegram_credentials() -> dict[str, str]
================================================================================
"""

import os


def resolve_credentials(exchange: str, testnet: bool = False) -> dict[str, str]:
    """
    Resolve exchange API credentials from environment.
    
    Convention:
        - Live: LIGHTHOUSE_<EXCHANGE>_API_KEY, LIGHTHOUSE_<EXCHANGE>_API_SECRET
        - Testnet: LIGHTHOUSE_<EXCHANGE>_TESTNET_API_KEY, LIGHTHOUSE_<EXCHANGE>_TESTNET_API_SECRET
    
    Args:
        exchange: Exchange name 
        testnet: If True, use testnet credentials
    
    Returns:
        dict with "api_key" and "api_secret"
    
    Raises:
        RuntimeError: Credentials not found or incomplete
    """
    exchange_upper = exchange.upper()
    
    if testnet:
        key_env = f"LIGHTHOUSE_{exchange_upper}_TESTNET_API_KEY"
        secret_env = f"LIGHTHOUSE_{exchange_upper}_TESTNET_API_SECRET"
        mode = "testnet"
    else:
        key_env = f"LIGHTHOUSE_{exchange_upper}_API_KEY"
        secret_env = f"LIGHTHOUSE_{exchange_upper}_API_SECRET"
        mode = "live"
    
    api_key = os.environ.get(key_env)
    api_secret = os.environ.get(secret_env)
    
    if not api_key:
        raise RuntimeError(f"Missing {mode} API key: {key_env}")
    if not api_secret:
        raise RuntimeError(f"Missing {mode} API secret: {secret_env}")
    
    return {
        "api_key": api_key,
        "api_secret": api_secret,
    }


def check_credentials_not_shared(exchange: str) -> None:
    """
    Guard against live and testnet credentials being identical for a given exchange.

    Reads both env var pairs directly (does not require either to be fully set).
    No-op if either pair is partially or fully missing. Raises if both pairs are
    present and byte-identical (key AND secret) — running "testnet" against live
    credentials (or vice versa) is a silent way to trade real money by mistake.

    Args:
        exchange: Exchange name

    Raises:
        RuntimeError: both live and testnet credentials are set and identical
    """
    exchange_upper = exchange.upper()

    live_key_env = f"LIGHTHOUSE_{exchange_upper}_API_KEY"
    live_secret_env = f"LIGHTHOUSE_{exchange_upper}_API_SECRET"
    testnet_key_env = f"LIGHTHOUSE_{exchange_upper}_TESTNET_API_KEY"
    testnet_secret_env = f"LIGHTHOUSE_{exchange_upper}_TESTNET_API_SECRET"

    live_key = os.environ.get(live_key_env)
    live_secret = os.environ.get(live_secret_env)
    testnet_key = os.environ.get(testnet_key_env)
    testnet_secret = os.environ.get(testnet_secret_env)

    if not (live_key and live_secret and testnet_key and testnet_secret):
        return  # one or both pairs incomplete — nothing to compare

    if live_key == testnet_key and live_secret == testnet_secret:
        raise RuntimeError(
            f"Live and testnet credentials for '{exchange}' are identical — "
            f"({live_key_env}/{live_secret_env}) match "
            f"({testnet_key_env}/{testnet_secret_env}). "
            "Live and testnet credentials must differ. Refusing to start."
        )


def resolve_telegram_credentials() -> dict[str, str]:
    """
    Resolve Telegram bot credentials from environment.

    Convention:
        LIGHTHOUSE_TELEGRAM_BOT_TOKEN, LIGHTHOUSE_TELEGRAM_CHAT_ID

    Returns:
        dict with "bot_token" and "chat_id"

    Raises:
        RuntimeError: one or both env vars are missing, naming the missing var(s).
    """
    bot_token = os.environ.get("LIGHTHOUSE_TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("LIGHTHOUSE_TELEGRAM_CHAT_ID")

    missing = [
        name for name, value in (
            ("LIGHTHOUSE_TELEGRAM_BOT_TOKEN", bot_token),
            ("LIGHTHOUSE_TELEGRAM_CHAT_ID", chat_id),
        ) if not value
    ]
    if missing:
        raise RuntimeError(f"Missing Telegram credential(s): {', '.join(missing)}")

    return {
        "bot_token": bot_token,
        "chat_id": chat_id,
    }
