from http.client import HTTPException
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from geo_model import GeoModel
from household import Household
from store import Store
import json
from decimal import Decimal

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

model = GeoModel()

@app.get("/api/stores")
async def get_stores():
    stores = model.get_stores()
    return {"stores_json": stores}

@app.get("/api/agents")
async def get_agents():
    agents = model.agents
    return {"agents_json": agents}

@app.get("/api/households")
async def get_households():
    households = model.get_households().astype(str)
    households_json = households.to_dict(orient="records")
    # Return as JSON response
    return {"households_json": households_json}

@app.delete("/api/remove-store")
async def remove_store(store_name: str = Body(...)):
    # Find the index of the store with the given name 
    store_index = next((index for (index, check) in enumerate(model.stores) if store_name == check[2]), None)
    storelist_index = next((index for (index, check) in enumerate(model.stores_list) if store_name == check.name), None)
    
    # Check if the store was found
    if store_index is None:
        raise HTTPException(status_code=404, detail="Store not found")
    
    #remove from store (names displayed) and stores_list (actual model)
    model.stores.pop(store_index) 
    model.stores_list.pop(storelist_index)
    return {"removed_store": store_name, "store_json": model.stores}

@app.put("/api/reset")
async def reset_all():
    #currently only resets stores, do we want to reset steps too?
    model.reset_stores()
    return {"store_json": model.stores}

@app.get("/api/get-step-number")
async def get_step_number():
    step_number = model.schedule.steps
    return {"step_number": step_number}

@app.put("/api/step")
async def step():
    model.step()
    step_number = model.schedule.steps
    return {"step_number": step_number}