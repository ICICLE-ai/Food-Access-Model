from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from geo_model import GeoModel
from household import Household
from store import Store


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

@app.get("/api/households")
async def get_households():
    households = model.get_households()
    return {"households_json": households}

@app.post("api/remove-store")
async def remove_store(store):
    print(store)
    print(type(store))
    model.deregister_agent(store)