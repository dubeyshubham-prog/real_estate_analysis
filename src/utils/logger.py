import logging
from config.settings import Config

# Create logs folder if it doesn't exist
Config.LOGS_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)

    # Avoid duplicate logs
    if logger.handlers:
        return logger

    # Console handler - shows logs in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(Config.LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT, Config.LOG_DATE_FORMAT))

    # File handler - saves logs to app.log file
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setLevel(Config.LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT, Config.LOG_DATE_FORMAT))

    # Add both handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger