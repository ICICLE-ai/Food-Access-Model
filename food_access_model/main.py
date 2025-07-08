from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

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
    allow_origins=[
        "http://localhost:5173",      # Vite dev server
        "http://127.0.0.1:5173",     
        "http://localhost:3000",      # other React setups
        "http://127.0.0.1:3000",   
        "https://fass.pods.icicleai.tapis.io"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
