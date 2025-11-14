"""
Utility decorators for error handling and logging
"""
import functools
import logging
import streamlit as st
from typing import Callable, Any

logger = logging.getLogger(__name__)


def error_handler(show_error: bool = True) -> Callable:
    """
    Error handling decorator
    
    Args:
        show_error: Whether to show error in UI
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} failed: {str(e)}")
                if show_error:
                    st.error(f"Operation failed: {str(e)}")
                raise
        return wrapper
    return decorator


def log_execution(func: Callable) -> Callable:
    """
    Execution logging decorator
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger.info(f"Starting execution of {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} executed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} execution failed: {str(e)}")
            raise
    return wrapper
