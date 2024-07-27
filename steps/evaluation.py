import logging
from zenml import step
import pandas as pd
@step
def evaluate_model(df)->None:
    """
    Evaluates Model on ingested data
    Args:
        df: the ingested Data
    """
    pass