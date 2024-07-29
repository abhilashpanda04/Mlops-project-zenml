import logging
# from pandera.typing import DataFrame,Series
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataDevideStretegy,DataPreProcessingStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    ]:
    """
    cleans the data and devides it into train and test
    Args:
        df: pd.DataFrame
    Returns:
        x_train: Training Data
        x_test: Testing Data
        y_train: Training Labels
        y_test: Testing Labels
    """
    try:
        preprocess_stretegy=DataPreProcessingStrategy()
        datacleaning=DataCleaning(df,preprocess_stretegy)
        processed_data=datacleaning.handle_data()


        devide_stretegy=DataDevideStretegy()
        datacleaning=DataCleaning(processed_data,devide_stretegy)
        x_train,x_test,y_train,y_test=datacleaning.handle_data()
        logging.info("Datacleaning completed")
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e