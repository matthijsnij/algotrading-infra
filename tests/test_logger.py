"""
================================================================================
UNIT TESTS FOR LOGGER.PY

To run all tests:
    pytest tests/ -v

To run all tests in this file:
    pytest tests/test_logger.py -v

To run a specific test function:
    pytest tests/test_logger.py -v -k "test_function_name"

- v flag makes pytest verbose
- k flag allows you to run tests by substring matching
================================================================================
"""

################### IMPORTS ##########################

import logging
import pytest
import lighthouse.utils.logging as logger_module
from lighthouse.utils.logging import get_logger, init_logging, add_bot_file_handler, _DatedRotatingFileHandler

################### FIXTURES ##########################

@pytest.fixture(autouse=True)
def reset_logger_state():
    """
    Reset logger module state and root logger handlers before and after each test
    so tests are fully isolated from one another.
    """
    # ── Teardown from any previous test ──────────────────────────────────────
    logger_module._initialized = False
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)

    yield

    # ── Teardown after this test ──────────────────────────────────────────────
    logger_module._initialized = False
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)


################### TESTS ##########################

# get_logger raises RuntimeError before init_logging is called
def test_get_logger_raises_before_init(tmp_path):
    with pytest.raises(RuntimeError, match="init_logging\\(\\) must be called before get_logger"):
        get_logger("main")


# add_bot_file_handler raises RuntimeError before init_logging is called
def test_add_bot_file_handler_raises_before_init(tmp_path):
    with pytest.raises(RuntimeError, match="init_logging\\(\\) must be called before add_bot_file_handler"):
        add_bot_file_handler("ExampleBot")


# calling init_logging twice does not add duplicate root handlers
def test_init_logging_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "LOG_DIR", tmp_path)

    init_logging("algotrading")
    init_logging("algotrading")

    root = logging.getLogger()
    dated_handlers = [h for h in root.handlers if isinstance(h, _DatedRotatingFileHandler)]
    assert len(dated_handlers) == 1


# calling add_bot_file_handler twice for the same bot does not add duplicate handlers
def test_add_bot_file_handler_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "LOG_DIR", tmp_path)

    init_logging("algotrading")
    add_bot_file_handler("ExampleBot")
    add_bot_file_handler("ExampleBot")

    bot_logger = logging.getLogger("ExampleBot")
    dated_handlers = [h for h in bot_logger.handlers if isinstance(h, _DatedRotatingFileHandler)]
    assert len(dated_handlers) == 1
