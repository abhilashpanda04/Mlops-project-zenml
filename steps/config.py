from pydantic import BaseModel
from typing import Optional


class ModelNameConfig(BaseModel):
    """Model algorithm configuration."""

    model_config = {"protected_namespaces": ()}

    model_name: Optional[str] = "LinearRegression"
