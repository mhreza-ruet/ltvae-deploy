from pydantic import BaseModel, Field
from typing import Dict, List

class PredictRequest(BaseModel):
    smiles: str = Field(
        ...,
        description="Enter your SMILES string here",
        example="enter your SMILES string here"
    )

class PredictResponse(BaseModel):
    input: str
    valid: bool
    latent: List[float]
    properties: Dict[str, float]
    meta: Dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_config = {"protected_namespaces": ()}