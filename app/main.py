import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.models.loader import ModelRegistry
from app.routers import autocomplete, health

ModelRegistry.initialize()

app = FastAPI(
    title="AI Coding Assistant API",
    description="FastAPI backend for streaming Python code generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(health.router)
app.include_router(autocomplete.router)
