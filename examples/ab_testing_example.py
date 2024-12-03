"""Example usage of A/B testing framework."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from backend.services.api.ab_testing import (
    ABTestingFramework,
    ModelVariant,
    TestStatus
)

async def simulate_predictions(
    framework: ABTestingFramework,
    test_id: str,
    num_predictions: int = 1000
):
    """Simulate model predictions for A/B test.
    
    Args:
        framework: A/B testing framework
        test_id: Test identifier
        num_predictions: Number of predictions to simulate
    """
    # Simulate predictions with different performance characteristics
    for _ in range(num_predictions):
        variant = framework.select_variant(test_id)
        if variant is None:
            continue
            
        # Simulate metrics
        if variant.model_id == "model_a":
            # Control variant
            accuracy = np.random.normal(0.75, 0.05)
            latency = np.random.normal(100, 10)
        else:
            # Test variant
            accuracy = np.random.normal(0.78, 0.05)
            latency = np.random.normal(95, 10)
            
        metrics = {
            "accuracy": accuracy,
            "latency_ms": latency
        }
        
        framework.record_metrics(test_id, variant, metrics)
        await asyncio.sleep(0.01)  # Simulate time between predictions

def plot_results(results: Dict, save_path: Path):
    """Plot test results.
    
    Args:
        results: Test results
        save_path: Path to save plots
    """
    metrics = results.get("metrics", {})
    if not metrics:
        return
        
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
        
    for i, (metric, data) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Plot metric values
        x = ["Control", "Test"]
        y = [data["control_value"], data["test_value"]]
        
        ax.bar(x, y)
        ax.set_title(f"{metric} Comparison")
        ax.set_ylabel(metric)
        
        # Add significance annotation
        if data["significant"]:
            ax.text(
                0.5,
                max(y) * 1.1,
                f"Significant (p={data['p_value']:.4f})",
                ha="center"
            )
        else:
            ax.text(
                0.5,
                max(y) * 1.1,
                f"Not Significant (p={data['p_value']:.4f})",
                ha="center"
            )
            
    plt.tight_layout()
    plt.savefig(save_path / "test_results.png")
    plt.close()

async def main():
    # Create test directory
    test_dir = Path("test_results")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize framework
    framework = ABTestingFramework(test_dir)
    
    # Create test
    test = framework.create_test(
        control_model_id="model_a",
        control_version="v1",
        test_model_id="model_b",
        test_version="v1",
        traffic_split=0.5,
        duration_days=1
    )
    
    print(f"Created test: {test.test_id}")
    
    # Start test
    framework.start_test(test.test_id)
    print("Started test")
    
    # Simulate predictions
    print("Simulating predictions...")
    await simulate_predictions(framework, test.test_id)
    
    # Complete test
    framework.complete_test(test.test_id)
    print("Completed test")
    
    # Get and plot results
    results = framework.get_test_results(test.test_id)
    if results:
        print("\nTest Results:")
        for metric, data in results["metrics"].items():
            print(f"\n{metric}:")
            print(f"  Control: {data['control_value']:.4f}")
            print(f"  Test: {data['test_value']:.4f}")
            print(f"  Improvement: {data['relative_improvement']*100:.2f}%")
            print(f"  Significant: {data['significant']}")
            print(f"  P-value: {data['p_value']:.4f}")
            
        plot_results(results, test_dir)
        print(f"\nPlots saved to {test_dir}")

if __name__ == "__main__":
    asyncio.run(main())
