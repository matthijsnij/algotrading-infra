"""
================================================================================
UNIT TESTS FOR NOTIFICATIONS/TELEGRAM.PY

To run all tests in this file:
    pytest tests/test_notifications/test_telegram.py -v
================================================================================
"""

################### IMPORTS ##########################

from unittest.mock import MagicMock, patch

from lighthouse.notifications.telegram import TelegramChannel

################### HELPERS ##########################

def _ok_response(status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    return response

################### TESTS ##########################

# successful send posts once with chat_id/text payload, no retry
def test_send_success_posts_once():
    channel = TelegramChannel("token123", "chat456")
    with patch("lighthouse.notifications.telegram.requests.post", return_value=_ok_response()) as mock_post, \
         patch("lighthouse.notifications.telegram.time.sleep") as mock_sleep:
        channel.send("hello", "info")

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.telegram.org/bottoken123/sendMessage"
    assert kwargs["json"]["chat_id"] == "chat456"
    assert "[INFO] hello" == kwargs["json"]["text"]
    mock_sleep.assert_not_called()

# network exception on every attempt is swallowed, retries with fixed delays, never raises
def test_send_retries_on_exception_then_gives_up():
    channel = TelegramChannel("token123", "chat456")
    with patch("lighthouse.notifications.telegram.requests.post", side_effect=ConnectionError("down")) as mock_post, \
         patch("lighthouse.notifications.telegram.time.sleep") as mock_sleep:
        channel.send("hello", "warning")  # must not raise

    assert mock_post.call_count == 3  # initial attempt + 2 retries
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0].args[0] == 1.0
    assert mock_sleep.call_args_list[1].args[0] == 3.0

# a transient failure followed by success stops retrying early
def test_send_succeeds_after_one_retry():
    channel = TelegramChannel("token123", "chat456")
    with patch(
        "lighthouse.notifications.telegram.requests.post",
        side_effect=[ConnectionError("down"), _ok_response()],
    ) as mock_post, patch("lighthouse.notifications.telegram.time.sleep") as mock_sleep:
        channel.send("hello", "info")

    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1

# HTTP 429 honors the Retry-After header instead of the fixed delay
def test_send_429_honors_retry_after_header():
    channel = TelegramChannel("token123", "chat456")
    rate_limited = _ok_response(status_code=429)
    rate_limited.headers = {"Retry-After": "7"}

    with patch(
        "lighthouse.notifications.telegram.requests.post",
        side_effect=[rate_limited, _ok_response()],
    ) as mock_post, patch("lighthouse.notifications.telegram.time.sleep") as mock_sleep:
        channel.send("hello", "info")

    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(7.0)

# repeated 429s exhaust all retries without raising
def test_send_429_exhausts_retries_without_raising():
    channel = TelegramChannel("token123", "chat456")
    rate_limited = _ok_response(status_code=429)
    rate_limited.headers = {"Retry-After": "2"}

    with patch("lighthouse.notifications.telegram.requests.post", return_value=rate_limited) as mock_post, \
         patch("lighthouse.notifications.telegram.time.sleep") as mock_sleep:
        channel.send("hello", "info")  # must not raise

    assert mock_post.call_count == 3
