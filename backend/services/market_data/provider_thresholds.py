from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class LatencyThresholds:
    warning: float = 500    # ms
    error: float = 1000     # ms
    critical: float = 2000  # ms

@dataclass
class ErrorRateThresholds:
    warning: float = 5      # errors/minute
    error: float = 10       # errors/minute
    critical: float = 20    # errors/minute

@dataclass
class HealthThresholds:
    warning: float = 80     # percentage
    error: float = 60       # percentage
    critical: float = 40    # percentage

@dataclass
class CacheThresholds:
    warning: float = 70     # percentage
    error: float = 50       # percentage

@dataclass
class ProviderThresholds:
    latency: LatencyThresholds = field(default_factory=LatencyThresholds)
    error_rate: ErrorRateThresholds = field(default_factory=ErrorRateThresholds)
    health: HealthThresholds = field(default_factory=HealthThresholds)
    cache: CacheThresholds = field(default_factory=CacheThresholds)

class ThresholdManager:
    def __init__(self):
        self.provider_thresholds: Dict[str, ProviderThresholds] = {}
        self.default_thresholds = ProviderThresholds()

    def set_provider_thresholds(self, provider_id: str, thresholds: ProviderThresholds):
        """Set thresholds for a specific provider"""
        self.provider_thresholds[provider_id] = thresholds
        logger.info(f"Updated thresholds for provider {provider_id}")

    def get_provider_thresholds(self, provider_id: str) -> ProviderThresholds:
        """Get thresholds for a specific provider, falling back to defaults if not set"""
        return self.provider_thresholds.get(provider_id, self.default_thresholds)

    def remove_provider_thresholds(self, provider_id: str):
        """Remove custom thresholds for a provider"""
        self.provider_thresholds.pop(provider_id, None)
        logger.info(f"Removed custom thresholds for provider {provider_id}")

    def set_default_thresholds(self, thresholds: ProviderThresholds):
        """Set default thresholds for all providers"""
        self.default_thresholds = thresholds
        logger.info("Updated default thresholds")

    def to_dict(self) -> Dict:
        """Convert thresholds to dictionary for serialization"""
        return {
            "default": self._thresholds_to_dict(self.default_thresholds),
            "providers": {
                provider_id: self._thresholds_to_dict(thresholds)
                for provider_id, thresholds in self.provider_thresholds.items()
            }
        }

    def from_dict(self, data: Dict):
        """Load thresholds from dictionary"""
        if "default" in data:
            self.default_thresholds = self._dict_to_thresholds(data["default"])
        
        if "providers" in data:
            self.provider_thresholds = {
                provider_id: self._dict_to_thresholds(thresholds)
                for provider_id, thresholds in data["providers"].items()
            }

    @staticmethod
    def _thresholds_to_dict(thresholds: ProviderThresholds) -> Dict:
        """Convert a ProviderThresholds instance to dictionary"""
        return {
            "latency": {
                "warning": thresholds.latency.warning,
                "error": thresholds.latency.error,
                "critical": thresholds.latency.critical
            },
            "error_rate": {
                "warning": thresholds.error_rate.warning,
                "error": thresholds.error_rate.error,
                "critical": thresholds.error_rate.critical
            },
            "health": {
                "warning": thresholds.health.warning,
                "error": thresholds.health.error,
                "critical": thresholds.health.critical
            },
            "cache": {
                "warning": thresholds.cache.warning,
                "error": thresholds.cache.error
            }
        }

    @staticmethod
    def _dict_to_thresholds(data: Dict) -> ProviderThresholds:
        """Convert a dictionary to ProviderThresholds instance"""
        return ProviderThresholds(
            latency=LatencyThresholds(
                warning=data["latency"]["warning"],
                error=data["latency"]["error"],
                critical=data["latency"]["critical"]
            ),
            error_rate=ErrorRateThresholds(
                warning=data["error_rate"]["warning"],
                error=data["error_rate"]["error"],
                critical=data["error_rate"]["critical"]
            ),
            health=HealthThresholds(
                warning=data["health"]["warning"],
                error=data["health"]["error"],
                critical=data["health"]["critical"]
            ),
            cache=CacheThresholds(
                warning=data["cache"]["warning"],
                error=data["cache"]["error"]
            )
        )

    def save_to_file(self, filepath: str):
        """Save thresholds to a JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved thresholds to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save thresholds to {filepath}: {e}")

    def load_from_file(self, filepath: str):
        """Load thresholds from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.from_dict(data)
            logger.info(f"Loaded thresholds from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load thresholds from {filepath}: {e}")
