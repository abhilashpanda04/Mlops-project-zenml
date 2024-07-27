import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from the data_path
    """
    def __init__(self,data_path:str) -> None:
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from data_path
        """
        logging.info(f"Ingesing data from path: {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """
    ingesting data from datapath

    Args:
        data_path:Path to the data
    Returns:
        pd.DataFrame: the ingested Data
    """

    try:
        ingest_data = IngestData(data_path=data_path)
        df=ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data from path {data_path}: {e}")