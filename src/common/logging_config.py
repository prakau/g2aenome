import sys
import os
import yaml
from loguru import logger
from pathlib import Path

# Determine project root dynamically
# Assuming this file is in src/common/logging_config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_logging_config(config_path=None):
    """Loads logging configuration from the main YAML config file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "main_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: Logging configuration file not found at {config_path}. Using default logger settings.", file=sys.stderr)
        return {
            "level": "INFO",
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "log_file_path": None # Default to console only
        }

    try:
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        logging_conf = full_config.get("logging", {})
        
        # Ensure default values if some keys are missing
        default_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        
        return {
            "level": logging_conf.get("level", "INFO").upper(),
            "log_file_path": logging_conf.get("log_file_path"), # Can be None
            "rotation": logging_conf.get("rotation", "10 MB"),
            "retention": logging_conf.get("retention", "10 days"),
            "format": logging_conf.get("format", default_format)
        }
    except Exception as e:
        print(f"Error loading logging configuration from {config_path}: {e}. Using default logger settings.", file=sys.stderr)
        return {
            "level": "INFO",
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "log_file_path": None
        }

def setup_logging(config_path=None):
    """
    Configures Loguru logger based on settings from a YAML configuration file.
    Reads logging level, format, and optional file logging parameters.
    """
    # Remove default handler to prevent duplicate console logs if re-configuring
    logger.remove()

    log_conf = load_logging_config(config_path)

    # Add console logger
    logger.add(
        sys.stderr,
        level=log_conf["level"],
        format=log_conf["format"],
        colorize=True
    )

    # Add file logger if path is specified
    if log_conf["log_file_path"]:
        log_file_full_path = PROJECT_ROOT / log_conf["log_file_path"]
        
        # Ensure log directory exists
        log_file_full_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file_full_path,
            level=log_conf["level"],
            format=log_conf["format"],
            rotation=log_conf["rotation"],
            retention=log_conf["retention"],
            enqueue=True,  # For asynchronous logging, good for performance
            backtrace=True, # Show full stack trace for exceptions
            diagnose=True   # Show variable values in tracebacks (careful with sensitive data)
        )
        logger.info(f"File logging enabled. Log file: {log_file_full_path}")
    else:
        logger.info("File logging not configured. Logging to console only.")
        
    logger.info(f"Loguru logger configured. Level: {log_conf['level']}")
    return logger

# Example usage (typically called once at the start of your application)
# if __name__ == "__main__":
#     # This will look for configs/main_config.yaml relative to project root
#     custom_logger = setup_logging() 
#     custom_logger.debug("This is a debug message.")
#     custom_logger.info("This is an info message.")
#     custom_logger.warning("This is a warning message.")
#     custom_logger.error("This is an error message.")
#     try:
#         1 / 0
#     except ZeroDivisionError:
#         custom_logger.exception("Caught an exception!")

# To use this logger in other modules:
# from loguru import logger
# logger.info("Message from another module")
