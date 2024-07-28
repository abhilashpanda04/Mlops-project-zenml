from pandera.typing import DataFrame
import logging
from zenml import step
from src.model_development import Linearregression
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
@step
def train_model(
    x_train:DataFrame,
    x_test:DataFrame,
    y_train:DataFrame,
    y_test:DataFrame,
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
        if config.model_name == "Linearregression":
            model=Linearregression()
            trained_model=model.train(x_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {config.model_name} not supported yet")

    except Exception as e:
        logging.error(f"Error in model training error: {e}")
        raise e
    

    
    