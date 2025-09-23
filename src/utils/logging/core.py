"""
Core logging configuration and setup
"""

import os
import sys
import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from .formatters import ColorFormatter, JSONFormatter
from .handlers import GZipRotatingFileHandler, AsyncHttpHandler, PerformanceHandler

class LoggerManager:
    """
    Centralized logging management system for the Voice Cloning Tool
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.environment = os.getenv('VOICE_CLONE_ENV', 'development')
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)

        # Initialize handlers
        self.setup_logging()
        
        # Performance tracking
        self.perf_handler = PerformanceHandler()
        logging.getLogger().addHandler(self.perf_handler)

        self._initialized = True

    def setup_logging(self):
        """Configure logging based on environment"""
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'color': {
                    '()': ColorFormatter   # ✅ use the class directly
                },
                'json': {
                    '()': JSONFormatter    # ✅ use the class directly
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'color',
                    'stream': sys.stdout
                },
                'file': {
                    '()': GZipRotatingFileHandler,
                    'filename': self.log_dir / 'voice_clone.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'json'
                },
                'error_file': {
                    '()': GZipRotatingFileHandler,
                    'filename': self.log_dir / 'error.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'json'
                }
            },
            'loggers': {
                '': {  # Root logger
                    'handlers': ['console', 'file'],
                    'level': 'INFO' if self.environment == 'production' else 'DEBUG'
                },
                'VoiceClone': {
                    'handlers': ['console', 'file', 'error_file'],
                    'level': 'INFO',
                    'propagate': False
                },
                'TextProcessor': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': False
                },
                'AudioRecorder': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': False
                },
                'numba': {
                'level': 'WARNING',  # Set Numba to a higher level
                'handlers': ['console', 'file'],
                'propagate': False
                },
                'matplotlib': {
                    'level': 'WARNING', # Silence matplotlib as well
                    'handlers': ['console', 'file'],
                    'propagate': False
                },
                'requests': {
                    'level': 'WARNING', # Silence the requests library
                    'handlers': ['console', 'file'],
                    'propagate': False
                }
            }
        }

        # Add remote logging in production
        if self.environment == 'production':
            config['handlers']['remote'] = {
                '()': AsyncHttpHandler,
                'url': os.getenv('LOG_SERVER_URL', 'http://localhost:8000/logs'),
                'formatter': 'json'
            }
            config['loggers']['']['handlers'].append('remote')

        logging.config.dictConfig(config)


    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance with the specified name"""
        return logging.getLogger(name)

    def start_operation(self, operation: str):
        """Start timing an operation"""
        self.perf_handler.start_operation(operation)

    def end_operation(self, operation: str):
        """End timing an operation"""
        self.perf_handler.end_operation(operation)

    def get_performance_metrics(self, operation: str) -> Dict[str, Any]:
        """Get performance metrics for an operation"""
        return self.perf_handler.get_metrics(operation)

    def set_log_level(self, level: str):
        """Dynamically change log level"""
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {level}')
        logging.getLogger().setLevel(numeric_level)

# Global logger instance
logger_manager = LoggerManager()

def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger instance"""
    return logger_manager.get_logger(name)

# Example usage
if __name__ == "__main__":
    # Get loggers for different components
    voice_logger = get_logger('VoiceClone')
    processor_logger = get_logger('TextProcessor')
    recorder_logger = get_logger('AudioRecorder')

    # Test different log levels
    voice_logger.debug("Debug message")
    voice_logger.info("Info message")
    voice_logger.warning("Warning message")
    voice_logger.error("Error message")

    # Test performance tracking
    logger_manager.start_operation('test_operation')
    import time
    time.sleep(1)
    logger_manager.end_operation('test_operation')
    metrics = logger_manager.get_performance_metrics('test_operation')
    print(f"Performance metrics: {metrics}")