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
    stores = model.get_stores()
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
    print(store)
    print(type(store))
    model.stores.pop(-1) #Placeholder remove TODO find store in list and remove
    print(model.stores)
    return {"removed_store": store}

@app.post("/api/reset")
async def reset_all():
    return {"success"}