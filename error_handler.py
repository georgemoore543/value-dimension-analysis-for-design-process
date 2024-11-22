from typing import Optional, Callable, Any, Dict
import time
import logging
from functools import wraps
from openai import (
    APIError, 
    RateLimitError, 
    APIConnectionError, 
    AuthenticationError,
    BadRequestError
)

class APIErrorHandler:
    """Handles OpenAI API errors with proper retry logic and logging"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default retry settings
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 16.0
        
        # Configure which errors are retryable
        self.retryable_errors = {
            RateLimitError: True,
            APIConnectionError: True,
            APIError: lambda e: e.status_code in {500, 502, 503, 504},
            BadRequestError: False,
            AuthenticationError: False
        }

    def is_retryable(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry"""
        for error_type, should_retry in self.retryable_errors.items():
            if isinstance(error, error_type):
                if callable(should_retry):
                    return should_retry(error)
                return should_retry
        return False

    def get_retry_delay(self, attempt: int, error: Exception) -> float:
        """Calculate delay before next retry using exponential backoff"""
        if isinstance(error, RateLimitError):
            # Use rate limit reset time if available
            reset_time = getattr(error, 'reset_time', None)
            if reset_time:
                return max(0, reset_time - time.time())
        
        # Default exponential backoff
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        # Add jitter to prevent thundering herd
        return delay * (0.5 + time.random())

    def handle_error(self, error: Exception, attempt: int) -> Optional[str]:
        """Handle API error and return error message if not retrying"""
        error_type = type(error).__name__
        
        if isinstance(error, AuthenticationError):
            self.logger.error(f"Authentication failed: {str(error)}")
            return "API key is invalid or expired"
            
        elif isinstance(error, BadRequestError):
            self.logger.error(f"Bad request: {str(error)}")
            return f"Invalid request parameters: {str(error)}"
            
        elif isinstance(error, RateLimitError):
            self.logger.warning(f"Rate limit exceeded (attempt {attempt}): {str(error)}")
            return "Rate limit exceeded" if attempt >= self.max_retries else None
            
        elif isinstance(error, APIConnectionError):
            self.logger.warning(f"Connection error (attempt {attempt}): {str(error)}")
            return "Failed to connect to API" if attempt >= self.max_retries else None
            
        else:
            self.logger.error(f"Unexpected {error_type}: {str(error)}")
            return f"Unexpected error: {str(error)}"

def with_error_handling(func: Callable) -> Callable:
    """Decorator to add error handling to API calls"""
    error_handler = APIErrorHandler()
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        last_error = None
        
        for attempt in range(error_handler.max_retries):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                error_message = error_handler.handle_error(e, attempt)
                
                if error_message:
                    raise RuntimeError(error_message) from e
                    
                if not error_handler.is_retryable(e):
                    raise
                    
                delay = error_handler.get_retry_delay(attempt, e)
                time.sleep(delay)
        
        # If we get here, we've exhausted our retries
        raise RuntimeError(f"Failed after {error_handler.max_retries} attempts") from last_error
        
    return wrapper 