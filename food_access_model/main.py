from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import json

from food_access_model.api.routes import router as api_router
from food_access_model.config.logging_config import setup_logging

# Load environment variables from .env file
load_dotenv(override=True)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Starting the application...")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fass.pods.icicleai.tapis.io", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)