import pandas as pd
import logging
from zenml import step
from src.model_development import LinearRegression
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
@step
def train_model(
    x_train:pd.DataFrame,
    x_test:pd.DataFrame,
    y_train:pd.DataFrame,
    y_test:pd.DataFrame,
    config:ModelNameConfig,
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
        if config.model_name == "LinearRegression":
            model=LinearRegression()
            trained_model=model.train(x_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {config.model_name} not supported yet")

    except Exception as e:
        logging.error(f"Error in model training error: {e}")
        raise e
    

    
    