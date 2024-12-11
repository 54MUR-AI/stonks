class ProviderError(Exception):
    """Base exception for provider-related errors"""
    pass

class NoAvailableProviderError(Exception):
    """Raised when no healthy providers are available"""
    pass

class ProviderConnectionError(ProviderError):
    """Raised when unable to connect to a provider"""
    pass

class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out"""
    pass

class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded"""
    pass
