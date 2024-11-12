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

@app.post("/api/add-store")
async def add_store(store: StoreInput):
    store_data = {"name": store.name, "category": store.category, "latitude": store.latitude, "longitude": store.longitude}

    # Parse the JSON data from the request body
    name = store_data["name"]
    print(model.stores[0])
    print(model.stores_list[0])
    # checking stores and storelist for name
    store_exists_in_stores = any(name == store[2] for store in model.stores)
    store_exists_in_stores_list = any(name == store.name for store in model.stores_list)
    # # If the store name exists in either list, return an error message (should change to id later on but doing this for MVP)
    if store_exists_in_stores or store_exists_in_stores_list:
        raise HTTPException(status_code=409, detail=f"Store with name '{name}' already exists.")
    
    #convert latitude and longitude to a polygon
    geo = str(convert_centroid_to_polygon(store_data["latitude"], store_data["longitude"], store_data["category"]))
    #does id matter if some get deleted, like does some operation rely on them being contiguous?
    model.stores.append([store_data["category"], geo, store_data["name"]])
    #TODO: investigate if we could get messed up by the id if a store gets deleted and now the ids are the same
    model.stores_list.append(Store(model=model, id=len(model.stores) + 1, name=name, type=store_data["category"], geometry=geo))
    #TODO: need type checking like category = SPM and name not in store_list
    print(model.stores)
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

@app.get("/api/get-num-households")
async def get_num_households():
    num_households = len(model.households)
    return {"num_households": num_households}

@app.get("/api/get-num-stores")
async def get_num_stores():
    num_stores = len(model.stores)
    return {"num_stores": num_stores}