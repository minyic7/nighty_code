# src/dataminer/core/exceptions.py
"""Custom exceptions for DataMiner module"""

from typing import Optional, Dict, Any, List


class DataMinerError(Exception):
    """Base exception for all DataMiner errors"""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None
    ):
        super().__init__(message)
        self.details = details or {}
        self.stage = stage
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.stage:
            base_msg = f"[{self.stage}] {base_msg}"
        if self.details:
            base_msg += f" (Details: {self.details})"
        return base_msg


class ExtractionError(DataMinerError):
    """Errors during data extraction"""
    
    def __init__(
        self,
        message: str,
        schema_name: Optional[str] = None,
        confidence_score: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.schema_name = schema_name
        self.confidence_score = confidence_score


class ValidationError(DataMinerError):
    """Errors during schema validation"""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        field_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []
        self.field_name = field_name


class SchemaError(DataMinerError):
    """Errors with schema definition or processing"""
    
    def __init__(
        self,
        message: str,
        schema_name: Optional[str] = None,
        schema_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.schema_name = schema_name
        self.schema_errors = schema_errors or []


class RepositoryError(DataMinerError):
    """Errors during repository analysis or processing"""
    
    def __init__(
        self,
        message: str,
        repository_path: Optional[str] = None,
        file_count: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.repository_path = repository_path
        self.file_count = file_count


class ConfigurationError(DataMinerError):
    """Errors in configuration"""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key


class ResourceError(DataMinerError):
    """Resource-related errors (memory, time, etc.)"""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        limit_exceeded: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.limit_exceeded = limit_exceeded


class IntegrationError(DataMinerError):
    """Errors with LLM/MCP/Copilot integration"""
    
    def __init__(
        self,
        message: str,
        integration_type: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.integration_type = integration_type
        self.provider = provider


class CacheError(DataMinerError):
    """Cache-related errors"""
    
    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.cache_key = cache_key


class TimeoutError(DataMinerError):
    """Timeout errors"""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class QualityError(DataMinerError):
    """Quality threshold not met"""
    
    def __init__(
        self,
        message: str,
        confidence_score: Optional[float] = None,
        threshold: Optional[float] = None,
        missing_fields: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.confidence_score = confidence_score
        self.threshold = threshold
        self.missing_fields = missing_fields or []