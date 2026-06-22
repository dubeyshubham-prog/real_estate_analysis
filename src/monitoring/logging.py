import logging
from config.settings import Config


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured application logger."""
    Config.initialize_directories()

    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)
    logger.propagate = False

    # Reuse existing handlers when this module is imported more than once.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(Config.LOG_LEVEL)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        filename=Config.LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setLevel(Config.LOG_LEVEL)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
