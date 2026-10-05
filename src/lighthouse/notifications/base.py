"""
================================================================================
NOTIFICATION CHANNEL BASE CLASS
================================================================================
Abstract interface for a single alerting backend.

Classes:
    BaseNotificationChannel : abstract send(message, level) interface
================================================================================
"""

############ IMPORTS ##############

from abc import ABC, abstractmethod

############ CLASS ##############

class BaseNotificationChannel(ABC):
    """
    Abstract base class for a single notification backend.

    Subclasses must never raise from send() — a channel failure must not be
    allowed to propagate back into bot/monitor/supervisor threads.
    """

    @abstractmethod
    def send(self, message: str, level: str) -> None:
        """
        Deliver a single message through this channel.

        Args:
            message: fully formatted human-readable message.
            level:   severity label (e.g. "info", "warning", "critical") — channels
                    may use this for formatting but must not raise on unknown values.
        """
        raise NotImplementedError
