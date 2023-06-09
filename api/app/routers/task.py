from typing import List
from fastapi import APIRouter
from app.config import get_settings, artifacts
from app.utils import create_response
from pydantic import BaseModel
import numpy as np
import torch
from transformers import pipeline

settings = get_settings()

router = APIRouter(
    prefix="/tasks",
    tags=["batch", "analytics"],
    responses={404: {"description": "Not found"}},
)


class InferenceForm(BaseModel):
    sentences: List[str]


@router.post("/financial-sentiment")
def sentiment_analysis(info: InferenceForm) -> List[float]:
    sentiment_pipeline = pipeline("sentiment-analysis", model=artifacts["model"]["finbert"],
                                  tokenizer=artifacts["tokenizer"]["finbert"],
                                  device=artifacts["device"])
    predictions = sentiment_pipeline(info.sentences)
    return create_response(
        message="embedding constructed",
        data=predictions
    )


@router.post("/tweet-sentiment")
def sentiment_analysis(info: InferenceForm) -> List[float]:
    sentiment_pipeline = pipeline("sentiment-analysis", model=artifacts["model"]["bertweet"],
                                  tokenizer=artifacts["tokenizer"]["bertweet"],
                                  device=artifacts["device"])
    predictions = sentiment_pipeline(info.sentences)
    return create_response(
        message="embedding constructed",
        data=predictions
    )
