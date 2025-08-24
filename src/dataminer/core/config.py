# src/dataminer/core/config.py
"""Configuration management for DataMiner"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import os
from ..core.types import ProcessingMode, ExtractionConfig


@dataclass
class DataMinerConfig:
    """Main configuration for DataMiner client"""
    
    # Core settings
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    
    # Integration configurations
    llm_config: Dict[str, Any] = field(default_factory=dict)
    mcp_config: Dict[str, Any] = field(default_factory=dict)
    copilot_config: Dict[str, Any] = field(default_factory=dict)
    
    # Caching
    enable_result_cache: bool = True
    cache_directory: Optional[Path] = None
    cache_size_limit_mb: int = 1000
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    enable_detailed_logging: bool = False
    
    # Performance
    memory_limit_mb: int = 4096
    temp_directory: Optional[Path] = None
    cleanup_temp_files: bool = True
    
    # Repository processing
    default_ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", ".svn", ".hg",
        "node_modules", ".venv", "venv", ".env",
        "*.log", "*.tmp", ".DS_Store"
    ])
    
    @classmethod
    def from_file(cls, config_path: Path) -> "DataMinerConfig":
        """Load configuration from YAML file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataMinerConfig":
        """Create configuration from dictionary"""
        # Handle nested extraction config
        extraction_data = data.get('extraction', {})
        extraction_config = ExtractionConfig(**extraction_data)
        
        # Handle path conversions
        if 'cache_directory' in data and data['cache_directory']:
            data['cache_directory'] = Path(data['cache_directory'])
        
        if 'log_file' in data and data['log_file']:
            data['log_file'] = Path(data['log_file'])
            
        if 'temp_directory' in data and data['temp_directory']:
            data['temp_directory'] = Path(data['temp_directory'])
        
        return cls(
            extraction=extraction_config,
            **{k: v for k, v in data.items() if k != 'extraction'}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {
            'extraction': {
                'default_mode': self.extraction.default_mode.value,
                'enable_caching': self.extraction.enable_caching,
                'cache_ttl_seconds': self.extraction.cache_ttl_seconds,
                'min_confidence_threshold': self.extraction.min_confidence_threshold,
                'max_gap_tolerance': self.extraction.max_gap_tolerance,
                'require_validation': self.extraction.require_validation,
                'max_concurrent_extractions': self.extraction.max_concurrent_extractions,
                'max_processing_time_seconds': self.extraction.max_processing_time_seconds,
                'chunk_size': self.extraction.chunk_size,
                'overlap_size': self.extraction.overlap_size,
                'preferred_provider': self.extraction.preferred_provider,
                'fallback_providers': self.extraction.fallback_providers,
                'model_settings': self.extraction.model_settings,
                'supported_languages': self.extraction.supported_languages,
                'max_repository_size_mb': self.extraction.max_repository_size_mb,
                'use_copilot_reasoning': self.extraction.use_copilot_reasoning,
                'enable_mcp_tools': self.extraction.enable_mcp_tools,
                'copilot_confidence_boost': self.extraction.copilot_confidence_boost,
            },
            'llm_config': self.llm_config,
            'mcp_config': self.mcp_config,
            'copilot_config': self.copilot_config,
            'enable_result_cache': self.enable_result_cache,
            'cache_directory': str(self.cache_directory) if self.cache_directory else None,
            'cache_size_limit_mb': self.cache_size_limit_mb,
            'log_level': self.log_level,
            'log_file': str(self.log_file) if self.log_file else None,
            'enable_detailed_logging': self.enable_detailed_logging,
            'memory_limit_mb': self.memory_limit_mb,
            'temp_directory': str(self.temp_directory) if self.temp_directory else None,
            'cleanup_temp_files': self.cleanup_temp_files,
            'default_ignore_patterns': self.default_ignore_patterns,
        }
        return result
    
    def save_to_file(self, config_path: Path):
        """Save configuration to YAML file"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate extraction settings
        if self.extraction.min_confidence_threshold < 0 or self.extraction.min_confidence_threshold > 1:
            issues.append("min_confidence_threshold must be between 0 and 1")
        
        if self.extraction.max_gap_tolerance < 0 or self.extraction.max_gap_tolerance > 1:
            issues.append("max_gap_tolerance must be between 0 and 1")
        
        if self.extraction.max_concurrent_extractions < 1:
            issues.append("max_concurrent_extractions must be at least 1")
        
        if self.extraction.chunk_size < 100:
            issues.append("chunk_size too small, minimum 100 tokens recommended")
        
        if self.extraction.overlap_size >= self.extraction.chunk_size:
            issues.append("overlap_size must be smaller than chunk_size")
        
        # Validate paths
        if self.cache_directory and not self.cache_directory.parent.exists():
            issues.append(f"Cache directory parent does not exist: {self.cache_directory.parent}")
        
        if self.log_file and not self.log_file.parent.exists():
            issues.append(f"Log file directory does not exist: {self.log_file.parent}")
        
        # Validate memory limits
        if self.memory_limit_mb < 512:
            issues.append("memory_limit_mb too low, minimum 512MB recommended")
        
        if self.cache_size_limit_mb > self.memory_limit_mb:
            issues.append("cache_size_limit_mb should not exceed memory_limit_mb")
        
        return issues


def create_default_config() -> DataMinerConfig:
    """Create a default configuration"""
    return DataMinerConfig(
        extraction=ExtractionConfig(
            default_mode=ProcessingMode.THOROUGH,
            preferred_provider="anthropic",
            fallback_providers=["openai"],
            min_confidence_threshold=0.7,
            use_copilot_reasoning=True,
            enable_mcp_tools=True
        ),
        llm_config={
            "temperature": 0.1,
            "max_tokens": 4000,
            "timeout": 60
        },
        mcp_config={
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "include_hidden": False
        },
        copilot_config={
            "enable_memory": True,
            "verbose": False,
            "auto_execute_tools": True
        },
        cache_directory=Path.home() / ".dataminer" / "cache",
        log_file=Path.home() / ".dataminer" / "logs" / "dataminer.log"
    )


def load_config_from_env() -> DataMinerConfig:
    """Load configuration from environment variables"""
    config = create_default_config()
    
    # Override with environment variables
    if os.getenv("DATAMINER_PROVIDER"):
        config.extraction.preferred_provider = os.getenv("DATAMINER_PROVIDER")
    
    if os.getenv("DATAMINER_CONFIDENCE_THRESHOLD"):
        config.extraction.min_confidence_threshold = float(os.getenv("DATAMINER_CONFIDENCE_THRESHOLD"))
    
    if os.getenv("DATAMINER_CACHE_DIR"):
        config.cache_directory = Path(os.getenv("DATAMINER_CACHE_DIR"))
    
    if os.getenv("DATAMINER_LOG_LEVEL"):
        config.log_level = os.getenv("DATAMINER_LOG_LEVEL")
    
    if os.getenv("DATAMINER_MEMORY_LIMIT"):
        config.memory_limit_mb = int(os.getenv("DATAMINER_MEMORY_LIMIT"))
    
    # Mode setting
    if os.getenv("DATAMINER_MODE"):
        mode_str = os.getenv("DATAMINER_MODE").lower()
        # Check against enum values, not members
        if mode_str in [mode.value for mode in ProcessingMode]:
            config.extraction.default_mode = ProcessingMode(mode_str)
    
    return config