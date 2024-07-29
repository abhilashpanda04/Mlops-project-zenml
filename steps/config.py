from pydantic import BaseModel
from typing import Optional

class ModelNameConfig(BaseModel):
    """Model configuration"""
    model_name:Optional[str]="LinearRegression"

model_config = {
"protected_namespaces": (),
"model_name": ModelNameConfig
}