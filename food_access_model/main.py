from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import json

from food_access_model.api.routes import router as api_router


def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

# Custom encoder for Decimal
class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        if not isinstance(obj, str):
            return str(obj)  # Or use str(obj) if you prefer strings
        return super().default(obj)


# Load environment variables from .env file
load_dotenv(override=True)

setup_logger()
logger = logging.getLogger(__name__)
logger.info("Starting the application...")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fass.pods.icicleai.tapis.io"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
