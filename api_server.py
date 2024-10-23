from fastapi import FastAPI, Body
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

@app.get("/api/stores")
async def get_stores():
    print(model.stores)
    stores = model.stores
    return {"stores": stores}

@app.get("/api/agents")
async def get_agents():
    agents = model.agents
    return {"agents_json": agents}

@app.get("/api/households")
async def get_households():
    households = model.get_households()
    return {"households_json": households}

@app.post("/api/remove-store")
async def remove_store(store: str = Body(...)):
    model.stores.pop(-1) #Placeholder for remove: TODO find store in list and remove correct store
    print(model.stores)
    return {"removed_store": store}

@app.put("/api/reset")
async def reset_all():
    model.stores = model.reset_stores()
    return {"success"}