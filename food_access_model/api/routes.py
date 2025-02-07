import logging

from fastapi import APIRouter, Depends, status

from food_access_model.abm.geo_model import GeoModel

from .helpers import StoreInput, convert_centroid_to_polygon



router = APIRouter(prefix="/api", tags=["ABM"])

model = GeoModel()

@router.get("/api/stores")
async def get_stores():
    stores = model.get_stores()
    return {"stores_json": stores}

@router.get("/api/agents")
async def get_agents():
    agents = model.agents
    return {"agents_json": agents}

@router.get("/api/households")
async def get_households():
    households = model.get_households().astype(str)
    households_json = households.to_dict(orient="records")
    # Return as JSON response
    return {"households_json": households_json}

@router.delete("/api/remove-store")
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

@router.put("/api/reset")
async def reset_all():
    #currently only resets stores, do we want to reset steps too?
    model.reset_stores()
    return {"store_json": model.stores}

@router.post("/api/add-store")
async def add_store(store: StoreInput):
    store_data = {"name": store.name, "category": store.category, "latitude": store.latitude, "longitude": store.longitude}

    # Parse the JSON data from the request body
    name = store_data["name"]
    # checking stores and storelist for name
    store_exists_in_stores = any(name == store[2] for store in model.stores)
    store_exists_in_stores_list = any(name == store.name for store in model.stores_list)
    # # If the store name exists in either list, return an error message (should change to id later on but doing this for MVP)
    if store_exists_in_stores or store_exists_in_stores_list:
        raise HTTPException(status_code=409, detail=f"Store with name '{name}' already exists.")
    
    #convert latitude and longitude to a polygon
    geo = str(convert_centroid_to_polygon(store_data["latitude"], store_data["longitude"], store_data["category"]))
    #does id matter if some get deleted, like does some operation rely on them being contiguous?
    model.stores.routerend([store_data["category"], geo, store_data["name"]])
    #TODO: investigate if we could get messed up by the id if a store gets deleted and now the ids are the same
    model.stores_list.routerend(Store(model=model, id=len(model.stores) + 1, name=name, type=store_data["category"], geometry=geo))
    #TODO: need type checking like category = SPM and name not in store_list
    return {"store_json": model.stores}

@router.get("/api/get-step-number")
async def get_step_number():
    step_number = model.schedule.steps
    return {"step_number": step_number}

@router.put("/api/step")
async def step():
    model.step()
    step_number = model.schedule.steps
    return {"step_number": step_number}

@router.get("/api/get-num-households")
async def get_num_households():
    num_households = len(model.households)
    return {"num_households": num_households}

@router.get("/api/get-num-stores")
async def get_num_stores():
    num_stores = len(model.stores)
    stores = model.stores_list
    numSPM = 0
    numNonSPM = 0
    for store in stores:
        if store is None:
            continue
        elif store.type == "supermarket" or store.type == "greengrocer" or store.type == "grocery":
            numSPM += 1
        else:
            numNonSPM += 1
    return {"num_stores": num_stores, "numSPM": numSPM, "numNonSPM": numNonSPM}

@router.get("/api/get-household-stats")
async def get_household_stats():
    households = model.households
    income = 0
    vehicles = 0
    for house in households:
        income += house[2]
        vehicles += house[4]
    avg_income = float(income)/len(households)
    avg_vehicles = float(vehicles)/len(households)
    return {"avg_income": avg_income, "avg_vehicles": avg_vehicles}
