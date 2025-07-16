from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import json
from contextlib import asynccontextmanager
import logging

from food_access_model.api.routes import router as api_router
from food_access_model.repository.db_repository import DBRepository, get_db_repository
from food_access_model.model_multi_processing.superclusters import initialize_supercluster_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom encoder for Decimal
class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        if not isinstance(obj, str):
            return str(obj)  # Or use str(obj) if you prefer strings
        return super().default(obj)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    db_repository = await get_db_repository()
   
    initialize_supercluster_cache(db_repository)
    logger.info("Geometry Supercluster cache initialized")
    
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Food Access Model API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
