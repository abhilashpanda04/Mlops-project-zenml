from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluation import evaluate_model

@pipeline
def train_pipeline(data_path:str,name:str) -> None:
    df=ingest_data(data_path)
    x_train,x_test,y_train,y_test=clean_data(df)
    model=train_model(x_train,x_test,y_train,y_test,name)
    r2_score,rmse=evaluate_model(model,x_test,y_test)
