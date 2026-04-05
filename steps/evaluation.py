import logging

import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step, ArtifactConfig, add_tags
from sklearn.base import RegressorMixin

from src.evaluation import MSE, R2, RMSE
from tag_registry import ArtifactType, Performance


@step
def evaluate_model(
    model: RegressorMixin,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[
        float,
        ArtifactConfig(
            name="r2_score",
            tags=[ArtifactType.METRIC.value],
        ),
    ],
    Annotated[
        float,
        ArtifactConfig(
            name="rmse",
            tags=[ArtifactType.METRIC.value],
        ),
    ],
]:
    """
    Evaluates model on test data and dynamically tags based on performance.

    Computes MSE, R2, and RMSE metrics. Dynamically applies performance
    tags to the trained model artifact based on evaluation thresholds.

    Args:
        model: Trained model
        x_test: Testing feature data
        y_test: Testing target data
    Returns:
        r2_score: R-squared score of the model
        rmse: Root mean squared error of the model
    """
    try:
        y_pred = model.predict(x_test)

        # Calculate all metrics
        mse_calculator = MSE()
        mse_score = mse_calculator.calculate_score(y_test, y_pred)

        r2_calculator = R2()
        r2_score = r2_calculator.calculate_score(y_test, y_pred)

        rmse_calculator = RMSE()
        rmse_score = rmse_calculator.calculate_score(y_test, y_pred)

        # Dynamically tag the trained model based on performance
        if r2_score >= 0.7:
            add_tags(tags=[Performance.HIGH_R2.value], artifact_name="trained_model", infer_artifact=True)
            logging.info(f"Model tagged as HIGH_R2 (R2={r2_score:.4f})")
        else:
            add_tags(tags=[Performance.LOW_R2.value], artifact_name="trained_model", infer_artifact=True)
            logging.info(f"Model tagged as LOW_R2 (R2={r2_score:.4f})")

        if rmse_score <= 1.0:
            add_tags(tags=[Performance.LOW_RMSE.value], artifact_name="trained_model", infer_artifact=True)
        else:
            add_tags(tags=[Performance.HIGH_RMSE.value], artifact_name="trained_model", infer_artifact=True)

        logging.info(f"Evaluation complete — R2: {r2_score:.4f} | RMSE: {rmse_score:.4f} | MSE: {mse_score:.4f}")
        return r2_score, rmse_score

    except Exception as e:
        logging.error(f"Error occurred while evaluating model: {e}")
        raise e
