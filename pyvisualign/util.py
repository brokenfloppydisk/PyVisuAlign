import time
import logging
from functools import wraps
from typing import Callable, TypeVar, Any

F = TypeVar('F', bound=Callable[..., Any])

def profile(func: F) -> F:
    """Decorator to profile function execution time.
    
    Logs the function name and execution time in milliseconds at DEBUG level.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000.0
            logging.debug(f"{func.__name__} completed in {elapsed:.3f} ms")
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000.0
            logging.debug(f"{func.__name__} failed after {elapsed:.3f} ms")
            raise
    return wrapper  # type: ignore
