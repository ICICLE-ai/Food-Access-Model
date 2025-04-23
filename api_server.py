import json
import os

from fastapi import Body, FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from food_access_model.api.routes import router as api_router
from food_access_model.api.routes import _run_model_step

from repository.db_repository import DBRepository, get_db_repository

from profiling import register_middlewares
import decimal
import shapely
import time


# Custom encoder for Decimal
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return str(obj)  # Convert Decimal to string
        
        if isinstance(obj, shapely.geometry.Polygon):
            return obj.wkt  # Convert Polygon to WKT (Well-Known Text)
        
        if isinstance(obj, shapely.geometry.Point):
            return obj.wkt  # Convert Point to WKT (Well-Known Text)
        
        if isinstance(obj, shapely.geometry.LineString):
            return obj.wkt  # Convert LineString to WKT (Well-Known Text)
        
        if isinstance(obj, shapely.geometry.MultiPolygon):
            return obj.wkt  # Convert MultiPolygon to WKT (Well-Known Text)
        
        if isinstance(obj, shapely.geometry.MultiPoint):
            return obj.wkt  # Convert MultiPoint to WKT (Well-Known Text)
        
        if isinstance(obj, shapely.geometry.MultiLineString):
            return obj.wkt  # Convert MultiLineString to WKT (Well-Known Text)
        
        return super().default(obj)




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown logic."""
    # Establish database connections
    try:
        start_time = time.time()
        
        repo = DBRepository()
        await repo.initialize()
        _run_model_step(repo)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Startup time: {duration:.2f} seconds", flush=True)
        yield  # The app runs while inside this context

    finally:
        print("Cleanup completed.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fassfrontstage.pods.icicleai.tapis.io"],  # React dev server. Used * for dev purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#register_middlewares(app)


app.include_router(api_router)