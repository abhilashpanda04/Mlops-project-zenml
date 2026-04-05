from zenml import pipeline, Model
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluation import evaluate_model
from tag_registry import Environment, Domain, PipelineType, ModelAlgorithm, Status


# Define Model entity for tracking in ZenML Model Control Plane
review_model = Model(
    name="customer_review_predictor",
    description="Predicts customer review scores based on order and product features",
    tags=[
        ModelAlgorithm.LINEAR_REGRESSION.value,
        Domain.CUSTOMER_REVIEWS.value,
        Status.EXPERIMENTAL.value,
    ],
)


@pipeline(
    tags=[
        PipelineType.TRAINING.value,
        Domain.ECOMMERCE.value,
        Environment.DEV.value,
    ],
    model=review_model,
)
def train_pipeline(data_path: str, name: str) -> None:
    """End-to-end training pipeline for customer review score prediction.

    Orchestrates data ingestion, cleaning, training, and evaluation steps.

    Args:
        data_path: Path to the raw dataset CSV file.
        name: Name of the model algorithm to use for training.
    """
    df = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test, name)
    r2_score, rmse = evaluate_model(model, x_test, y_test)
