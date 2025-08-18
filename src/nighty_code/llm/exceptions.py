"""
Custom exceptions for LLM module.
"""


class LLMException(Exception):
    """Base exception for all LLM-related errors."""
    pass


class InputTooLargeError(LLMException):
    """Raised when input exceeds token limits."""
    
    def __init__(self, token_count: int, max_tokens: int, message: str = None):
        self.token_count = token_count
        self.max_tokens = max_tokens
        if message is None:
            message = f"Input too large: {token_count} tokens exceeds maximum of {max_tokens}"
        super().__init__(message)


class OutputTruncatedError(LLMException):
    """Raised when output is truncated due to token limits."""
    
    def __init__(self, partial_response: str, reason: str = "max_tokens"):
        self.partial_response = partial_response
        self.reason = reason
        message = f"Output truncated due to {reason}. Partial response available."
        super().__init__(message)


class TokenLimitExceededError(LLMException):
    """Raised when total tokens (input + output) exceed model limits."""
    
    def __init__(self, total_tokens: int, max_tokens: int):
        self.total_tokens = total_tokens
        self.max_tokens = max_tokens
        message = f"Total tokens ({total_tokens}) exceeds model limit ({max_tokens})"
        super().__init__(message)


class APIError(LLMException):
    """Raised when API call fails."""
    
    def __init__(self, provider: str, error_code: str = None, message: str = None):
        self.provider = provider
        self.error_code = error_code
        full_message = f"API error from {provider}"
        if error_code:
            full_message += f" ({error_code})"
        if message:
            full_message += f": {message}"
        super().__init__(full_message)


class ConfigurationError(LLMException):
    """Raised when LLM configuration is invalid."""
    pass