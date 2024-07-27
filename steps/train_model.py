import pandas as pd
import logging
from zenml import step

@step
def train_model(df)->None:
    """
    Trains the model on ingested data
    Args:
        df: the ingested Data
    """
    pass