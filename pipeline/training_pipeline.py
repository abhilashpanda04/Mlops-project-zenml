from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluation import evaluate_model

@pipeline
def train_pipeline(data_path:str) -> None:
    df=ingest_data(data_path)
    df=clean_data(df)
    train_model(df)
    evaluate_model(df)
