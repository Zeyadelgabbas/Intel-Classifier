from pydantic import BaseModel
from typing import List


class PredictionResponse(BaseModel):
    base_name : str
    class_index: int
    class_name : str 
    confidence : float 


class PredictionsResponse(BaseModel):
    predictions: List[PredictionResponse]