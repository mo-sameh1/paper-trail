import logging
import sys

# Define color codes for ANSI escaping
COLORS = {
    "DEBUG": "\\033[94m",  # Light Blue
    "INFO": "\\033[92m",  # Light Green
    "WARNING": "\\033[93m",  # Light Yellow
    "ERROR": "\\033[91m",  # Light Red
    "CRITICAL": "\\033[1;91m",  # Bold Red
}
RESET = "\\033[0m"


class ColorFormatter(logging.Formatter):
    """
    Format logs with ANSI color codes based on severity level.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = (
            f"{COLORS.get(record.levelname, '')}%(asctime)s - "
            f"%(name)s - %(levelname)s - %(message)s{RESET}"
        )
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes to disk after every emit."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a configured logger instance that outputs to BOTH the console
    (with colors) and a standard file ('pipeline.log').
    """
    logger = logging.getLogger(name)

    # Avoid attaching duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console Handler (Colorized)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter())
        logger.addHandler(console_handler)

        # File Handler (Plaintext)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = FlushingFileHandler("pipeline.log", encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
