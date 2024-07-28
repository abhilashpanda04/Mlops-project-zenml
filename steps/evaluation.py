import logging
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE,R2,RMSE
from typing_extensions import Annotated
from typing import Tuple
from pandera.typing import DataFrame

@step
def evaluate_model(model:RegressorMixin,
                   x_test:DataFrame,
                   y_test:DataFrame)->Tuple[Annotated[float,"r2_score"],
                                               Annotated[float,"rmse"]]:
    """
    Evaluates Model on ingested data
    Args:
        model: Trained model
    """
    try:
        y_pred=model.predict(x_test)
        mse=MSE()
        mse_score=mse.calculate_score(y_test,y_pred)

        r2=R2()
        r2_scorer=r2.calculate_score(y_test,y_pred)

        rmse=RMSE()
        rmse_score=rmse.calculate_score(y_test,y_pred)

        return r2_scorer,rmse_score
    except Exception as e:
        logging.error(f"Error occured while evaluating model {e}")
        raise e


    