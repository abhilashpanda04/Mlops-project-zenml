import logging

import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step, ArtifactConfig, add_tags

from src.data_cleaning import DataCleaning, DataDevideStretegy, DataPreProcessingStrategy
from tag_registry import ArtifactType, Domain, DataQuality


@step
def clean_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[
        pd.DataFrame,
        ArtifactConfig(
            name="x_train",
            tags=[ArtifactType.PROCESSED.value, Domain.ECOMMERCE.value],
        ),
    ],
    Annotated[
        pd.DataFrame,
        ArtifactConfig(
            name="x_test",
            tags=[ArtifactType.PROCESSED.value, Domain.ECOMMERCE.value],
        ),
    ],
    Annotated[
        pd.Series,
        ArtifactConfig(
            name="y_train",
            tags=[ArtifactType.PROCESSED.value, Domain.CUSTOMER_REVIEWS.value],
        ),
    ],
    Annotated[
        pd.Series,
        ArtifactConfig(
            name="y_test",
            tags=[ArtifactType.PROCESSED.value, Domain.CUSTOMER_REVIEWS.value],
        ),
    ],
]:
    """
    Cleans the data and divides it into train and test sets.

    Applies preprocessing strategy to handle missing values and
    feature selection, then splits into training and test sets.
    Dynamically tags artifacts based on data quality assessment.

    Args:
        df: Raw input DataFrame
    Returns:
        x_train: Training feature data
        x_test: Testing feature data
        y_train: Training target labels
        y_test: Testing target labels
    """
    try:
        # Assess data quality before cleaning
        missing_pct = df.isnull().mean().mean() * 100
        logging.info(f"Missing data percentage: {missing_pct:.2f}%")

        # Preprocessing
        preprocess_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        processed_data = data_cleaning.handle_data()

        # Dynamically tag based on data quality
        if missing_pct == 0:
            add_tags(tags=[DataQuality.COMPLETE.value], artifact_name="x_train", infer_artifact=True)
            add_tags(tags=[DataQuality.COMPLETE.value], artifact_name="x_test", infer_artifact=True)
        else:
            add_tags(tags=[DataQuality.INCOMPLETE.value], artifact_name="x_train", infer_artifact=True)
            add_tags(tags=[DataQuality.INCOMPLETE.value], artifact_name="x_test", infer_artifact=True)

        # Train-test split
        devide_strategy = DataDevideStretegy()
        data_cleaning = DataCleaning(processed_data, devide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()

        logging.info(f"Data cleaning completed. Train size: {len(x_train)}, Test size: {len(x_test)}")
        return x_train, x_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e
