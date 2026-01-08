import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Configure logging for the application.
    
    Logging level is controlled by the LOG_LEVEL environment variable.
    Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
    Default: INFO (for production)
    
    Environment Variables:
        LOG_LEVEL: Set the logging level (default: INFO)
        LOG_FILE: Path to log file (default: app.log)
    
    Logging Strategy:
    - logging.info(): Used for significant runtime events (service start/stop, 
      key user actions, important state changes) relevant for operational monitoring
      in production.
    - logging.debug(): Used for verbose output intended for development 
      (variable values, detailed execution flow, non-critical events).
    """
    
    # Get logging level from environment, default to INFO for production
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "app.log")
    
    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level_str not in valid_levels:
        log_level_str = "INFO"
    
    log_level = getattr(logging, log_level_str)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Define format for logging
    log_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)
    
    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(log_format)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    return root_logger