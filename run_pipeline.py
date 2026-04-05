"""
Main entry point for running the training pipeline.

Demonstrates:
- Running pipelines with YAML configuration
- Adding runtime tags via with_options
- Using tag registry for consistent tagging
"""

from pipeline.training_pipeline import train_pipeline
from tag_registry import ModelAlgorithm, Environment


if __name__ == "__main__":
    # Run pipeline with YAML config + runtime tags
    configured_pipeline = train_pipeline.with_options(
        config_path="configuration/config.yaml",
        tags=[
            ModelAlgorithm.LINEAR_REGRESSION.value,
            "run-type-manual",
        ],
    )
    configured_pipeline()
