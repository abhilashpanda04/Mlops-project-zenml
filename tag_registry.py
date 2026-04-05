"""
Tag Registry for consistent tagging across the project.

Uses Python Enums to avoid typos and ensure consistency.
All tags used in pipelines, artifacts, and models should be defined here.
"""

from enum import Enum


class Environment(Enum):
    """Environment tags to track where a pipeline/model is running."""
    DEV = "environment-development"
    STAGING = "environment-staging"
    PRODUCTION = "environment-production"


class Domain(Enum):
    """Domain tags to categorize the business area."""
    ECOMMERCE = "domain-ecommerce"
    CUSTOMER_REVIEWS = "domain-customer-reviews"


class PipelineType(Enum):
    """Pipeline type tags to classify pipeline purpose."""
    TRAINING = "pipeline-training"
    INFERENCE = "pipeline-inference"
    EVALUATION = "pipeline-evaluation"


class ArtifactType(Enum):
    """Artifact type tags to categorize data artifacts."""
    RAW = "artifact-raw"
    PROCESSED = "artifact-processed"
    FEATURE = "artifact-feature"
    MODEL = "artifact-model"
    METRIC = "artifact-metric"


class ModelAlgorithm(Enum):
    """Algorithm tags to identify which model was used."""
    LINEAR_REGRESSION = "algorithm-linear-regression"


class Status(Enum):
    """Status tags to track lifecycle stage."""
    EXPERIMENTAL = "status-experimental"
    VALIDATED = "status-validated"
    PRODUCTION = "status-production"


class DataQuality(Enum):
    """Data quality tags applied dynamically based on data checks."""
    COMPLETE = "quality-complete"
    INCOMPLETE = "quality-incomplete"
    HIGH_QUALITY = "quality-high"
    LOW_QUALITY = "quality-low"


class Performance(Enum):
    """Model performance tags applied dynamically based on evaluation."""
    HIGH_R2 = "performance-high-r2"
    LOW_R2 = "performance-low-r2"
    HIGH_RMSE = "performance-high-rmse"
    LOW_RMSE = "performance-low-rmse"
