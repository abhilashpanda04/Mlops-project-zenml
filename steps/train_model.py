import logging

import pandas as pd
from typing_extensions import Annotated
from zenml import step, ArtifactConfig
from sklearn.pipeline import Pipeline

from src.model_development import LinearRegressionModel
from tag_registry import ArtifactType, ModelAlgorithm


@step
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    name: str,
) -> Annotated[
    Pipeline,
    ArtifactConfig(
        name="trained_model",
        tags=[
            ArtifactType.MODEL.value,
            ModelAlgorithm.LINEAR_REGRESSION.value,
        ],
    ),
]:
    """
    Trains the model on ingested data.

    Uses the Strategy pattern via src/model_development.py to support
    multiple model algorithms. Currently supports LinearRegression.

    Args:
        x_train: Training feature data
        x_test: Testing feature data
        y_train: Training target data
        y_test: Testing target data
        name: Model algorithm name (e.g., "LinearRegression")
    Returns:
        Trained model instance
    """
    try:
        model = None
        if name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(x_train, y_train)
            logging.info(f"Model '{name}' training completed successfully")
            return trained_model
        else:
            raise ValueError(f"Model '{name}' not supported yet")

    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise e
