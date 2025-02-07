from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from geo_model import GeoModel
from household import Household
from store import Store
import json
from decimal import Decimal
from api_helper import StoreInput, convert_centroid_to_polygon


from food_access_model.api.routes import router as api_router

# Custom encoder for Decimal
class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        if not isinstance(obj, str):
            return str(obj)  # Or use str(obj) if you prefer strings
        return super().default(obj)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
