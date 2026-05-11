from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Football ML Prediction API")

# Autoriser Lovable à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Import des routers (on les créera après) ---
from routers import predict, data

app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
app.include_router(data.router, prefix="/data", tags=["Football Data"])

@app.get("/")
def root():
    return {"status": "✅ Football ML API en ligne"}

@app.get("/health")
def health():
    return {"status": "ok"}
