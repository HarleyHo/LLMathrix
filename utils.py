# utils.py
import logging


def setup_logging(log_file: str = "task.log") -> logging.Logger:
    """
    Configure logging for the application, saving logs to both console and a file.

    Args:
        log_file (str): Path to the log file. Defaults to 'task.log'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger instance
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # Set the logger's level
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger

    # Console handler for output to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    # File handler for output to a file
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Set httpx logger to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


class CompletionError(Exception):
    """Custom exception for completion generation failures."""
    pass
