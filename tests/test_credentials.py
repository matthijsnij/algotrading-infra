"""
================================================================================
UNIT TESTS FOR CONFIG/CREDENTIALS.PY

To run all tests in this file:
    pytest tests/test_credentials.py -v
================================================================================
"""

################### IMPORTS ##########################

import pytest

from lighthouse.config.credentials import check_credentials_not_shared, resolve_telegram_credentials

################### HELPERS ##########################

ENV_VARS = [
    "LIGHTHOUSE_PHEMEX_API_KEY",
    "LIGHTHOUSE_PHEMEX_API_SECRET",
    "LIGHTHOUSE_PHEMEX_TESTNET_API_KEY",
    "LIGHTHOUSE_PHEMEX_TESTNET_API_SECRET",
    "LIGHTHOUSE_TELEGRAM_BOT_TOKEN",
    "LIGHTHOUSE_TELEGRAM_CHAT_ID",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure none of the credential env vars leak in from the real environment."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)

################### TESTS ##########################

# no-op when only the live pair is set
def test_no_op_when_only_live_pair_set(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_API_KEY", "key")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_API_SECRET", "secret")
    check_credentials_not_shared("phemex")  # should not raise

# no-op when only the testnet pair is set
def test_no_op_when_only_testnet_pair_set(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_TESTNET_API_KEY", "key")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_TESTNET_API_SECRET", "secret")
    check_credentials_not_shared("phemex")  # should not raise

# no-op when both pairs are set but differ
def test_no_op_when_both_set_and_different(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_API_KEY", "live_key")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_API_SECRET", "live_secret")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_TESTNET_API_KEY", "testnet_key")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_TESTNET_API_SECRET", "testnet_secret")
    check_credentials_not_shared("phemex")  # should not raise

# raises when both pairs are set and byte-identical
def test_raises_when_both_set_and_identical(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_API_KEY", "same_key")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_API_SECRET", "same_secret")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_TESTNET_API_KEY", "same_key")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_TESTNET_API_SECRET", "same_secret")
    with pytest.raises(RuntimeError, match="identical"):
        check_credentials_not_shared("phemex")

# no-op when one pair is only partially set (missing secret)
def test_no_op_when_pair_partially_missing(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_API_KEY", "same_key")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_API_SECRET", "same_secret")
    monkeypatch.setenv("LIGHTHOUSE_PHEMEX_TESTNET_API_KEY", "same_key")
    # testnet secret intentionally left unset
    check_credentials_not_shared("phemex")  # should not raise


# ── resolve_telegram_credentials() ──────────────────────────────────────────

# both env vars set, returns dict with bot_token/chat_id
def test_resolve_telegram_credentials_success(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_TELEGRAM_BOT_TOKEN", "tok-123")
    monkeypatch.setenv("LIGHTHOUSE_TELEGRAM_CHAT_ID", "chat-456")

    creds = resolve_telegram_credentials()

    assert creds == {"bot_token": "tok-123", "chat_id": "chat-456"}

# missing bot token raises RuntimeError naming the missing var
def test_resolve_telegram_credentials_missing_token(monkeypatch):
    monkeypatch.delenv("LIGHTHOUSE_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("LIGHTHOUSE_TELEGRAM_CHAT_ID", "chat-456")

    with pytest.raises(RuntimeError, match="LIGHTHOUSE_TELEGRAM_BOT_TOKEN"):
        resolve_telegram_credentials()

# missing chat id raises RuntimeError naming the missing var
def test_resolve_telegram_credentials_missing_chat_id(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_TELEGRAM_BOT_TOKEN", "tok-123")
    monkeypatch.delenv("LIGHTHOUSE_TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(RuntimeError, match="LIGHTHOUSE_TELEGRAM_CHAT_ID"):
        resolve_telegram_credentials()

# both missing raises RuntimeError naming both vars
def test_resolve_telegram_credentials_both_missing(monkeypatch):
    monkeypatch.delenv("LIGHTHOUSE_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("LIGHTHOUSE_TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(RuntimeError, match="LIGHTHOUSE_TELEGRAM_BOT_TOKEN.*LIGHTHOUSE_TELEGRAM_CHAT_ID"):
        resolve_telegram_credentials()
