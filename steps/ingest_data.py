import logging

import pandas as pd
from typing import Annotated
from zenml import step, ArtifactConfig

from tag_registry import ArtifactType, Domain


class IngestData:
    """
    Ingesting the data from the data_path
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
        Ingesting data from data_path
        """
        logging.info(f"Ingesting data from path: {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> Annotated[
    pd.DataFrame,
    ArtifactConfig(
        name="raw_customer_data",
        tags=[
            ArtifactType.RAW.value,
            Domain.ECOMMERCE.value,
        ],
    ),
]:
    """
    Ingesting data from datapath

    Args:
        data_path: Path to the data
    Returns:
        pd.DataFrame: the ingested Data
    """
    try:
        ingest = IngestData(data_path=data_path)
        df = ingest.get_data()
        logging.info(f"Ingested {len(df)} rows from {data_path}")
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data from path {data_path}: {e}")
        raise e
