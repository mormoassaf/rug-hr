from app import config
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.routers import task
from mangum import Mangum
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="BERT as a service")

settings = config.get_settings()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.whitelist)
app.include_router(task.router)


def load_artifacts():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"{device} is available")
    config.artifacts["device"] = device
    config.artifacts["tokenizer"]["finbert"] = AutoTokenizer.from_pretrained(
        "./app/artifacts/tokenizers/finbert")
    config.artifacts["model"]["finbert"] = AutoModelForSequenceClassification.from_pretrained(
        "./app/artifacts/models/finbert").to(device)
    config.artifacts["tokenizer"]["bertweet"] = AutoTokenizer.from_pretrained(
        "./app/artifacts/tokenizers/bertweet")
    config.artifacts["model"]["bertweet"] = AutoModelForSequenceClassification.from_pretrained(
        "./app/artifacts/models/bertweet").to(device)


@app.on_event("startup")
def load_prerequisites():
    config.log.info("=== BERT as a Service ===")
    config.log.info("\tloading models...")
    load_artifacts()
    config.log.info("\tloading complete...")


@app.get("/info")
async def info(settings: config.Settings = Depends(config.get_settings)):
    return {
        "app_name": settings.app_name,
        "description": settings.description,
    }


handler = Mangum(app)
