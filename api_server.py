from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from geo_model import GeoModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = GeoModel()

@app.get("/api/data")
async def get_data():
    return {"message": "Hello from FastAPI"}

@app.get("/api/stores")
async def get_stores():
    stores = model.get_stores()
    return {"stores": stores}

@app.get("api/agents")
async def get_agents():
    agents = model.agents
    return {"agents_json": agents}