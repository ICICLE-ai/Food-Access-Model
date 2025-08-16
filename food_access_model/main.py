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
        "https://fass.pods.icicleai.tapis.io",
        "https://fassfrontstage.pods.icicleai.tapis.io",
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://127.0.0.1:5173",  # Alternative localhost for Vite
        "http://localhost:8080",  # Vue/other dev servers
        "http://127.0.0.1:8080",  # Alternative localhost
        "*",  # Allow all origins for development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(api_router)
