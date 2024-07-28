import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression
import pandas as pd
class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self,x_train,y_train):
        """
        train the model
        Args:
            x_train: Training data
            y_train: Target data

        Returns:
            None
        """
        
class Linearregression(Model):
    """
    Linear Regression model
    """
    def train(self,x_train:pd.DataFrame,y_train:pd.DataFrame,**kwargs):
        """
        Trains a model
        Args:
            x_train: Training data
            y_train: Target data

        Returns:
            None
        """
        try:
            reg=LinearRegression(**kwargs)
            reg.fit(x_train,y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error in model training error: {e}")

        