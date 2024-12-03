"""A/B testing framework for model evaluation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from enum import Enum
import logging
import json
from pathlib import Path
import asyncio
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """A/B test status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ModelVariant:
    """Model variant configuration."""
    model_id: str
    version: str
    traffic_percentage: float
    metrics: Dict[str, float] = None
    sample_count: int = 0

@dataclass
class ABTest:
    """A/B test configuration and results."""
    test_id: str
    control_variant: ModelVariant
    test_variant: ModelVariant
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TestStatus = TestStatus.PENDING
    metrics: Dict[str, Dict[str, float]] = None
    significance_level: float = 0.05

class ABTestingFramework:
    """Framework for managing A/B tests."""
    
    def __init__(self, storage_path: Path):
        """Initialize framework.
        
        Args:
            storage_path: Path to store test results
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_tests: Dict[str, ABTest] = {}
        self._load_active_tests()
        
    def _load_active_tests(self):
        """Load active tests from storage."""
        try:
            for test_file in self.storage_path.glob("*.json"):
                with open(test_file, "r") as f:
                    test_data = json.load(f)
                    test = self._deserialize_test(test_data)
                    if test.status == TestStatus.RUNNING:
                        self.active_tests[test.test_id] = test
                        
        except Exception as e:
            logger.error(f"Error loading active tests: {str(e)}")
            
    def _serialize_test(self, test: ABTest) -> dict:
        """Serialize test to dictionary.
        
        Args:
            test: A/B test
            
        Returns:
            Serialized test
        """
        return {
            "test_id": test.test_id,
            "control_variant": {
                "model_id": test.control_variant.model_id,
                "version": test.control_variant.version,
                "traffic_percentage": test.control_variant.traffic_percentage,
                "metrics": test.control_variant.metrics,
                "sample_count": test.control_variant.sample_count
            },
            "test_variant": {
                "model_id": test.test_variant.model_id,
                "version": test.test_variant.version,
                "traffic_percentage": test.test_variant.traffic_percentage,
                "metrics": test.test_variant.metrics,
                "sample_count": test.test_variant.sample_count
            },
            "start_time": test.start_time.isoformat(),
            "end_time": test.end_time.isoformat() if test.end_time else None,
            "status": test.status.value,
            "metrics": test.metrics,
            "significance_level": test.significance_level
        }
        
    def _deserialize_test(self, data: dict) -> ABTest:
        """Deserialize test from dictionary.
        
        Args:
            data: Serialized test
            
        Returns:
            Deserialized test
        """
        return ABTest(
            test_id=data["test_id"],
            control_variant=ModelVariant(**data["control_variant"]),
            test_variant=ModelVariant(**data["test_variant"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            status=TestStatus(data["status"]),
            metrics=data["metrics"],
            significance_level=data["significance_level"]
        )
        
    def create_test(
        self,
        control_model_id: str,
        control_version: str,
        test_model_id: str,
        test_version: str,
        traffic_split: float = 0.5,
        duration_days: int = 7,
        significance_level: float = 0.05
    ) -> ABTest:
        """Create new A/B test.
        
        Args:
            control_model_id: Control model ID
            control_version: Control model version
            test_model_id: Test model ID
            test_version: Test model version
            traffic_split: Percentage of traffic for test variant
            duration_days: Test duration in days
            significance_level: Statistical significance level
            
        Returns:
            Created test
        """
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        test = ABTest(
            test_id=test_id,
            control_variant=ModelVariant(
                model_id=control_model_id,
                version=control_version,
                traffic_percentage=1 - traffic_split
            ),
            test_variant=ModelVariant(
                model_id=test_model_id,
                version=test_version,
                traffic_percentage=traffic_split
            ),
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(days=duration_days),
            status=TestStatus.PENDING,
            significance_level=significance_level
        )
        
        # Save test configuration
        with open(self.storage_path / f"{test_id}.json", "w") as f:
            json.dump(self._serialize_test(test), f, indent=2)
            
        return test
        
    def start_test(self, test_id: str) -> bool:
        """Start A/B test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Success status
        """
        try:
            test_path = self.storage_path / f"{test_id}.json"
            if not test_path.exists():
                logger.error(f"Test {test_id} not found")
                return False
                
            with open(test_path, "r") as f:
                test = self._deserialize_test(json.load(f))
                
            test.status = TestStatus.RUNNING
            self.active_tests[test_id] = test
            
            # Update test file
            with open(test_path, "w") as f:
                json.dump(self._serialize_test(test), f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error starting test {test_id}: {str(e)}")
            return False
            
    def select_variant(self, test_id: str) -> Optional[ModelVariant]:
        """Select variant for request based on traffic split.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Selected variant
        """
        if test_id not in self.active_tests:
            return None
            
        test = self.active_tests[test_id]
        if test.status != TestStatus.RUNNING:
            return None
            
        # Random selection based on traffic split
        if random.random() < test.test_variant.traffic_percentage:
            return test.test_variant
        return test.control_variant
        
    def record_metrics(
        self,
        test_id: str,
        variant: ModelVariant,
        metrics: Dict[str, float]
    ):
        """Record metrics for variant.
        
        Args:
            test_id: Test identifier
            variant: Model variant
            metrics: Performance metrics
        """
        if test_id not in self.active_tests:
            return
            
        test = self.active_tests[test_id]
        if test.status != TestStatus.RUNNING:
            return
            
        # Update metrics
        if variant.model_id == test.control_variant.model_id:
            test.control_variant.metrics = metrics
            test.control_variant.sample_count += 1
        else:
            test.test_variant.metrics = metrics
            test.test_variant.sample_count += 1
            
        # Save updated test
        with open(self.storage_path / f"{test_id}.json", "w") as f:
            json.dump(self._serialize_test(test), f, indent=2)
            
    def evaluate_test(self, test_id: str) -> Dict[str, Dict[str, float]]:
        """Evaluate test results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test results
        """
        if test_id not in self.active_tests:
            return {}
            
        test = self.active_tests[test_id]
        
        # Calculate statistical significance
        results = {}
        for metric, control_value in test.control_variant.metrics.items():
            test_value = test.test_variant.metrics.get(metric)
            if test_value is None:
                continue
                
            # Perform t-test
            t_stat, p_value = self._calculate_significance(
                control_value,
                test_value,
                test.control_variant.sample_count,
                test.test_variant.sample_count
            )
            
            results[metric] = {
                "control_value": control_value,
                "test_value": test_value,
                "difference": test_value - control_value,
                "relative_improvement": (test_value - control_value) / control_value,
                "p_value": p_value,
                "significant": p_value < test.significance_level
            }
            
        return results
        
    def _calculate_significance(
        self,
        control_value: float,
        test_value: float,
        control_samples: int,
        test_samples: int
    ) -> tuple:
        """Calculate statistical significance.
        
        Args:
            control_value: Control metric value
            test_value: Test metric value
            control_samples: Number of control samples
            test_samples: Number of test samples
            
        Returns:
            Tuple of (t-statistic, p-value)
        """
        # Simplified t-test calculation
        # In practice, you would want to use scipy.stats.ttest_ind
        try:
            # Calculate pooled standard deviation
            s = np.sqrt(
                (control_value ** 2 / control_samples +
                 test_value ** 2 / test_samples) / 2
            )
            
            # Calculate t-statistic
            t_stat = (test_value - control_value) / (
                s * np.sqrt(1/control_samples + 1/test_samples)
            )
            
            # Calculate p-value (simplified)
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
            
            return t_stat, p_value
            
        except Exception:
            return 0, 1
            
    def _normal_cdf(self, x: float) -> float:
        """Calculate normal cumulative distribution function.
        
        Args:
            x: Input value
            
        Returns:
            CDF value
        """
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))
        
    def complete_test(self, test_id: str):
        """Complete A/B test.
        
        Args:
            test_id: Test identifier
        """
        if test_id not in self.active_tests:
            return
            
        test = self.active_tests[test_id]
        test.status = TestStatus.COMPLETED
        test.end_time = datetime.now()
        
        # Calculate final results
        test.metrics = self.evaluate_test(test_id)
        
        # Save final results
        with open(self.storage_path / f"{test_id}.json", "w") as f:
            json.dump(self._serialize_test(test), f, indent=2)
            
        # Remove from active tests
        del self.active_tests[test_id]
        
    def get_test_results(self, test_id: str) -> Optional[Dict]:
        """Get test results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test results
        """
        try:
            with open(self.storage_path / f"{test_id}.json", "r") as f:
                return json.load(f)
        except Exception:
            return None
