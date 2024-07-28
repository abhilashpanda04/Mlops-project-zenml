import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args: 
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            score: float
        """
        pass

class MSE(Evaluation):
    """
    Evaluation stretegy for mean squared error
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        caluclates MSE score
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            score: float
        """
        try:
            logging.info("Entered the calculate_score method of the MSE class")
            mse=mean_squared_error(y_true, y_pred)
            logging.info(f"MSE : {mse}")
            return mse
        except Exception as e:
            logging.error(f"error in calculating mse{e}")
            raise e
        
class R2(Evaluation):
    """
    Evaluation stretegy for R2 score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        caluclates R2 score
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            score: float
        """
        try:
            logging.info("calculating R2 score")
            r2=r2_score(y_true, y_pred)
            logging.info(f"R2 score is :{r2}")

        except Exception as e:
            logging.error(f"error in calculating R2 score {e}")
            raise e
        
class RMSE(Evaluation):
    """Evaluation stretegy for root mean squared error"""
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        caluclates RMSE score
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            score: float
        """
        try:
            logging.info("calculating RMSE score")
            rmse=mean_squared_error(y_true, y_pred,squared=False)
            logging.info(f"RMSE score is :{rmse}")

        except Exception as e:
            logging.error(f"error in calculating RMSE score {e}")
            raise e
