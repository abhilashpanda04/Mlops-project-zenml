# from pandera.typing import DataFrame
import pandas as pd
import logging
from zenml import step
from src.model_development import LinearRegressionModel
from sklearn.base import RegressorMixin
@step
def train_model(
    x_train:pd.DataFrame,
    x_test:pd.DataFrame,
    y_train:pd.DataFrame,
    y_test:pd.DataFrame,
    name,
    )->RegressorMixin:
    """
    Trains the model on ingested data
    Args:
        x_train: Training data
        x_test: Testing data
        y_train: Target data
        y_test: Testing target
    """
    try:
        model=None
        if name == "LinearRegression":
            model=LinearRegressionModel()
            trained_model=model.train(x_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {config.model_name} not supported yet")

    except Exception as e:
        logging.error(f"Error in model training error: {e}")
        raise e
    

    
    