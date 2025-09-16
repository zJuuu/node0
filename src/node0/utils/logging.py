# Copyright 2025 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from logging.handlers import RotatingFileHandler

from hivemind.utils import get_logger, use_hivemind_log_handler
from hivemind.utils.logging import TextStyle, always_log_caller


class CustomFormatter(logging.Formatter):
    """
    A formatter that allows a log time and caller info to be overridden via
    ``logger.log(level, message, extra={"origin_created": ..., "caller": ...})``.
    """

    _LEVEL_TO_COLOR = {
        logging.DEBUG: TextStyle.PURPLE,
        logging.INFO: TextStyle.BLUE,
        logging.WARNING: TextStyle.ORANGE,
        logging.ERROR: TextStyle.RED,
        logging.CRITICAL: TextStyle.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "origin_created"):
            record.created = record.origin_created
            record.msecs = (record.created - int(record.created)) * 1000

        if record.levelno > logging.INFO or always_log_caller:
            if not hasattr(record, "caller"):
                record.caller = f"{record.name}.{record.funcName}:{record.lineno}"
            record.caller_block = f" [{TextStyle.BOLD}{record.caller}{TextStyle.RESET}]"
        else:
            record.caller_block = ""

        # Aliases for the format argument
        record.levelcolor = (
            self._LEVEL_TO_COLOR[record.levelno] if record.levelno in self._LEVEL_TO_COLOR else TextStyle.BLUE
        )
        record.bold = TextStyle.BOLD
        record.reset = TextStyle.RESET

        return super().format(record)


class Node0Logger:
    def __init__(self, log_level: str = "INFO"):
        """Instantiate logger.

        Args:
            log_level (str, optional): logging level. Defaults to "INFO".
            log_file (str | None, optional): file to save logs. Defaults to None.
        """
        # Add extra log level
        EXTRA_LEVEL = 15  # between INFO (20) and DEBUG (10)
        logging.addLevelName(EXTRA_LEVEL, "EXTRA")

        # Create a custom method for the logger
        def extra(self, message, *args, **kwargs):
            if self.isEnabledFor(EXTRA_LEVEL):
                self._log(EXTRA_LEVEL, message, args, **kwargs)

        logging.Logger.extra = extra
        use_hivemind_log_handler("in_root_logger")

        # Convert log level string to logging constant
        numeric_level = getattr(logging, log_level.upper(), log_level)

        # Configure root logger
        self.root_logger = get_logger()
        self.root_logger.setLevel(numeric_level)

        formatter = CustomFormatter(
            fmt="{asctime}.{msecs:03.0f} [{bold}{levelcolor}{levelname}{reset}]{caller_block} {message}",
            style="{",
            datefmt="%b %d %H:%M:%S",
        )

        for handler in list(self.root_logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                handler.setFormatter(formatter)

    def add_monitor_handler(self, monitor_handler: logging.Handler):
        """Attach monitor handler to report logs."""
        self.root_logger.addHandler(monitor_handler)
