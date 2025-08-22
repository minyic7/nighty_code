# src/llm/core/config.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from .types import LLMProvider, LLMConfig
from ..middleware.rate_limiter import RateLimitConfig


@dataclass
class PoolConfig:
    """Configuration for the LLM connection pool"""
    min_size: int = 1
    max_size: int = 10
    acquire_timeout: float = 10.0
    idle_timeout: float = 3600.0  # 1 hour
    max_lifetime: float = 7200.0  # 2 hours
    retry_on_error: bool = True
    health_check_interval: float = 60.0
    enable_metrics: bool = True


@dataclass
class GlobalConfig:
    """Global configuration for the LLM module"""
    default_provider: Optional[LLMProvider] = None
    providers: Dict[LLMProvider, LLMConfig] = field(default_factory=dict)
    pool_config: PoolConfig = field(default_factory=PoolConfig)
    enable_logging: bool = True
    log_level: str = "INFO"
    metrics_enabled: bool = True


class ConfigManager:
    """Manages LLM configurations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.global_config = GlobalConfig()
        self.provider_configs: List[LLMConfig] = []
        
        # Load configuration from YAML file
        if config_path is None:
            # Look for config file in standard locations
            possible_paths = [
                Path("config/llm.yaml"),
                Path("../../../config/llm.yaml"),  # When running from src/llm/core
                Path(__file__).parent.parent.parent.parent / "config" / "llm.yaml",
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            self._load_from_yaml(config_path)
        else:
            # Fallback to environment variables if no config file
            self._load_from_env()
    
    def _load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load provider configurations
        for provider_name, provider_config in config.get('providers', {}).items():
            try:
                provider = LLMProvider(provider_name)
                api_keys = provider_config.get('api_keys', [])
                models = provider_config.get('models', [])
                settings = provider_config.get('settings', {})
                
                # Assert equal number of keys and models
                if api_keys and models:
                    if len(api_keys) != len(models):
                        raise ValueError(
                            f"Provider {provider_name} must have equal number of api_keys ({len(api_keys)}) "
                            f"and models ({len(models)})"
                        )
                
                # Load rate limit configs
                rate_limit_configs = []
                if 'rate_limits' in provider_config:
                    rate_limits = provider_config['rate_limits']
                    
                    # Handle both single dict (all keys use same limits) and list (per-key limits)
                    if isinstance(rate_limits, dict):
                        # Single rate limit config for all keys
                        rate_limit_config = RateLimitConfig(
                            requests_per_minute=rate_limits.get('requests_per_minute', 50),
                            input_tokens_per_minute=rate_limits.get('input_tokens_per_minute', 50000),
                            output_tokens_per_minute=rate_limits.get('output_tokens_per_minute', 10000),
                            total_tokens_per_minute=rate_limits.get('total_tokens_per_minute')
                        )
                        # Use same config for all keys
                        rate_limit_configs = [rate_limit_config] * len(api_keys)
                    
                    elif isinstance(rate_limits, list):
                        # Per-key rate limit configs
                        if len(rate_limits) != len(api_keys):
                            raise ValueError(
                                f"Provider {provider_name} must have equal number of api_keys ({len(api_keys)}) "
                                f"and rate_limits ({len(rate_limits)})"
                            )
                        
                        for rl in rate_limits:
                            rate_limit_configs.append(RateLimitConfig(
                                requests_per_minute=rl.get('requests_per_minute', 50),
                                input_tokens_per_minute=rl.get('input_tokens_per_minute', 50000),
                                output_tokens_per_minute=rl.get('output_tokens_per_minute', 10000),
                                total_tokens_per_minute=rl.get('total_tokens_per_minute')
                            ))
                else:
                    # No rate limits specified
                    rate_limit_configs = [None] * len(api_keys)
                
                # Create a config for each API key with its corresponding model and rate limits
                for i, api_key in enumerate(api_keys):
                    if api_key:  # Skip empty keys
                        model = models[i] if i < len(models) else models[0]
                        rate_limit_config = rate_limit_configs[i] if i < len(rate_limit_configs) else None
                        
                        self.provider_configs.append(
                            LLMConfig(
                                provider=provider,
                                api_key=api_key,
                                model=model,
                                base_url=settings.get('base_url'),
                                temperature=settings.get('temperature', 0.7),
                                max_tokens=settings.get('max_tokens'),
                                timeout=settings.get('timeout', 30),
                                max_retries=settings.get('max_retries', 3),
                                rate_limit_config=rate_limit_config
                            )
                        )
                        # Add the first valid config to global config (for backward compatibility)
                        if provider not in self.global_config.providers:
                            self.global_config.providers[provider] = self.provider_configs[-1]
            except ValueError as e:
                # Re-raise config errors
                if "must have equal number" in str(e):
                    raise
                # Skip unknown providers
                continue
        
        # Load pool configuration
        pool_config = config.get('pool', {})
        self.global_config.pool_config = PoolConfig(
            min_size=pool_config.get('min_size', 1),
            max_size=pool_config.get('max_size', 10),
            acquire_timeout=pool_config.get('acquire_timeout', 10.0),
            idle_timeout=pool_config.get('idle_timeout', 3600.0),
            max_lifetime=pool_config.get('max_lifetime', 7200.0),
            retry_on_error=pool_config.get('retry_on_error', True),
            health_check_interval=pool_config.get('health_check_interval', 60.0),
            enable_metrics=pool_config.get('enable_metrics', True),
        )
        
        # Load global settings
        global_config = config.get('global', {})
        if global_config.get('default_provider'):
            try:
                self.global_config.default_provider = LLMProvider(global_config['default_provider'])
            except ValueError:
                pass
        self.global_config.enable_logging = global_config.get('enable_logging', True)
        self.global_config.log_level = global_config.get('log_level', 'INFO')
        self.global_config.metrics_enabled = global_config.get('metrics_enabled', True)
    
    def _load_from_env(self):
        """Load configuration from environment variables (fallback)"""
        # OpenAI configuration
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.add_provider(
                LLMConfig(
                    provider=LLMProvider.OPENAI,
                    api_key=api_key,
                    model=os.getenv("OPENAI_MODEL", "gpt-4"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "0")) or None,
                )
            )
        
        # Anthropic configuration
        if api_key := os.getenv("ANTHROPIC_API_KEY"):
            self.add_provider(
                LLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    api_key=api_key,
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
                    base_url=os.getenv("ANTHROPIC_BASE_URL"),
                    temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
                    max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "0")) or None,
                )
            )
        
        # Pool configuration
        self.global_config.pool_config.min_size = int(
            os.getenv("LLM_POOL_MIN_SIZE", "1")
        )
        self.global_config.pool_config.max_size = int(
            os.getenv("LLM_POOL_MAX_SIZE", "10")
        )
    
    def add_provider(self, config: LLMConfig):
        """Add or update a provider configuration"""
        self.global_config.providers[config.provider] = config
        if not self.global_config.default_provider:
            self.global_config.default_provider = config.provider
    
    def get_provider_config(self, provider: LLMProvider) -> Optional[LLMConfig]:
        """Get configuration for a specific provider"""
        return self.global_config.providers.get(provider)
    
    def get_all_provider_configs(self, provider: LLMProvider) -> List[LLMConfig]:
        """Get all configurations for a specific provider (for multiple API keys)"""
        return [cfg for cfg in self.provider_configs if cfg.provider == provider]
    
    def set_default_provider(self, provider: LLMProvider):
        """Set the default provider"""
        if provider not in self.global_config.providers:
            raise ValueError(f"Provider {provider} not configured")
        self.global_config.default_provider = provider
    
    def get_default_config(self) -> Optional[LLMConfig]:
        """Get the default provider configuration"""
        if self.global_config.default_provider:
            return self.get_provider_config(self.global_config.default_provider)
        return None


# Global configuration instance
config_manager = ConfigManager()