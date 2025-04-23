import logging

from fastapi import APIRouter, Body, HTTPException, Depends

#from ..abm.geo_model import GeoModel

#from .helpers import StoreInput, convert_centroid_to_polygon

#from ..abm.store import Store



from food_access_model.api.helpers import StoreInput, convert_centroid_to_polygon
from food_access_model.abm.geo_model import GeoModel
from model_multi_processing.batch_running import batch_run
from food_access_model.abm.store import Store
from repository.db_repository import DBRepository, get_db_repository
from food_access_model.abm.geo_model import GeoModel


router = APIRouter(prefix="/api", tags=["ABM"])

#model = GeoModel()

#FRONT_URL = os.environ.get("FRONT_URL", "http://localhost:5173")



@router.get("/stores")
async def get_stores(repository: DBRepository = Depends(get_db_repository)):
    model = repository.get_model()
    stores = model.get_stores()
    return {"stores_json": stores}


@router.get("/agents")
async def get_agents(repository: DBRepository = Depends(get_db_repository)):
    model = repository.get_model()
    agents = model.agents
    return {"agents_json": agents}


@router.get("/households")
async def get_households(repository: DBRepository = Depends(get_db_repository)):
    model = repository.get_model()
    households = model.get_households().astype(str)
    households_json = households.to_dict(orient="records")
    # Return as JSON response
    return {"households_json": households_json}


@router.delete("/remove-store")
async def remove_store(store_name: str = Body(...), repository: DBRepository = Depends(get_db_repository)):
    model = repository.get_model()
    # Find the index of the store with the given name
    store_index = next(
        (index for (index, check) in enumerate(model.stores) if store_name == check[2]),
        None,
    )
    storelist_index = next(
        (
            index
            for (index, check) in enumerate(model.stores_list)
            if store_name == check.name
        ),
        None,
    )

    # Check if the store was found
    if store_index is None:
        raise HTTPException(status_code=404, detail="Store not found")

    # remove from store (names displayed) and stores_list (actual model)
    model.stores.pop(store_index)
    model.stores_list.pop(storelist_index)
    return {"removed_store": store_name, "store_json": model.stores}


@router.put("/reset")
async def reset_all(repository: DBRepository = Depends(get_db_repository)):
    model = repository.get_model()
    # currently only resets stores, do we want to reset steps too?
    model.reset_stores()
    return {"store_json": model.stores}


@router.post("/add-store")
async def add_store(store: StoreInput, repository: DBRepository = Depends(get_db_repository)):
    model = repository.get_model()
    store_data = {
        "name": store.name,
        "category": store.category,
        "latitude": store.latitude,
        "longitude": store.longitude,
    }

    # Parse the JSON data from the request body
    name = store_data["name"]
    # checking stores and storelist for name
    store_exists_in_stores = any(name == store[2] for store in model.stores)
    store_exists_in_stores_list = any(name == store.name for store in model.stores_list)
    # # If the store name exists in either list, return an error message (should change to id later on but doing this for MVP)
    if store_exists_in_stores or store_exists_in_stores_list:
        raise HTTPException(
            status_code=409, detail=f"Store with name '{name}' already exists."
        )

    # convert latitude and longitude to a polygon
    geo = str(
        convert_centroid_to_polygon(
            store_data["latitude"], store_data["longitude"], store_data["category"]
        )
    )
    # does id matter if some get deleted, like does some operation rely on them being contiguous?
    model.stores.append([store_data["category"], geo, store_data["name"]])
    # TODO: investigate if we could get messed up by the id if a store gets deleted and now the ids are the same
    model.stores_list.append(
        Store(
            model=model,
            id=len(model.stores) + 1,
            name=name,
            type=store_data["category"],
            geometry=geo,
        )
    )
    # TODO: need type checking like category = SPM and name not in store_list
    return {"store_json": model.stores}


@router.get("/get-step-number")
async def get_step_number(repository: DBRepository = Depends(get_db_repository)):
    model = repository.get_model()
    step_number = model.schedule.steps
    return {"step_number": step_number}

def filter_unique_by_runid(results):
    unique_results = {}
    for item in results:
        # If the item is a dictionary, use it directly.
        if isinstance(item, dict):
            run_id = item.get("RunId")
            if run_id is not None and run_id not in unique_results:
                unique_results[run_id] = item
        # If the item is a list of dictionaries, iterate over it.
        elif isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, dict):
                    run_id = sub_item.get("RunId")
                    if run_id is not None and run_id not in unique_results:
                        unique_results[run_id] = sub_item
    return list(unique_results.values())

def _run_model_step(repository: DBRepository):
    stores = repository.get_food_stores()
    households = repository.get_households()    

    print(f"Model begining", flush=True)
    
    results = batch_run(
        GeoModel,
        parameters={
            "households": households,
            "stores": stores,
        },
        max_steps=1,
        data_collection_period=1,
        display_progress=True,
        number_processes=60,
    )
    
    print(f"Model Step Completed", flush=True)
        
    print("Running model update", flush=True)
    
    #Implementation Miguel r
    all_households = []
    finalResults = filter_unique_by_runid(results)
    
    
    for res in results:
        # If res is a dictionary, extract data directly.
        if isinstance(res, dict):
            #all_stores.extend(res.get("stores", []))
            all_households.extend(res.get("households", []))
            
    
        elif isinstance(res, list):
             for item in res:
                 if isinstance(item, dict):
                     all_households.extend(item.get("households", []))
                     

    # print(f" Total Households Before: {len(repository.get_households())}", flush=True)
    # print(f" Total stores Before: {len(repository.get_food_stores())}", flush=True)
    allStores =  finalResults[0].get("stores", [])


    repository.update_model(all_households, allStores)
    #print("Model update completed", flush=True)
    step_number = repository.get_model().raw_step_number
    return step_number

@router.put("/step")
async def step(repository: DBRepository = Depends(get_db_repository)):

    """ step_number = await _run_model_step(repository) """
    step_number =  _run_model_step(repository)

    return {"step_number": step_number}


@router.get("/get-num-households")
async def get_num_households(repository: DBRepository = Depends(get_db_repository)):
    model = repository.get_model()
    num_households = len(model.households)
    return {"num_households": num_households}


@router.get("/get-num-stores")
async def get_num_stores(repository: DBRepository = Depends(get_db_repository)):
    model = repository.get_model()
    num_stores = len(model.stores)
    stores = model.stores_list
    numSPM = 0
    numNonSPM = 0
    for store in stores:
        if store is None:
            continue
        elif (
            store.type == "supermarket"
            or store.type == "greengrocer"
            or store.type == "grocery"
        ):
            numSPM += 1
        else:
            numNonSPM += 1
    return {"num_stores": num_stores, "numSPM": numSPM, "numNonSPM": numNonSPM}


@router.get("/get-household-stats")
async def get_household_stats(repository: DBRepository = Depends(get_db_repository)):
    
    model = repository.get_model()
    households = model.households
    income = 0
    vehicles = 0
    for house in households:
        income += house[2]
        vehicles += house[4]
    avg_income = float(income) / len(households)
    avg_vehicles = float(vehicles) / len(households)
    return {"avg_income": avg_income, "avg_vehicles": avg_vehicles}





""" @router.get("/stores")
async def get_stores():
    stores = model.get_stores()
    return {"stores_json": stores}

@router.get("/agents")
async def get_agents():
    agents = model.agents
    return {"agents_json": agents}

@router.get("/households")
async def get_households():
    households = model.get_households().astype(str)
    households_json = households.to_dict(orient="records")
    # Return as JSON response
    return {"households_json": households_json}

@router.delete("/remove-store")
async def remove_store(store_name: str = Body(...)):
    # Find the index of the store with the given name
    store_index = next(
        (index for (index, check) in enumerate(model.stores) if store_name == check[2]),
        None,
    )
    storelist_index = next(
        (
            index
            for (index, check) in enumerate(model.stores_list)
            if store_name == check.name
        ),
        None,
    )

    # Check if the store was found
    if store_index is None:
        raise HTTPException(status_code=404, detail="Store not found")

    # remove from store (names displayed) and stores_list (actual model)
    model.stores.pop(store_index)
    model.stores_list.pop(storelist_index)
    return {"removed_store": store_name, "store_json": model.stores}

@router.put("/reset")
async def reset_all():
    # currently only resets stores, do we want to reset steps too?
    model.reset_stores()
    return {"store_json": model.stores}

@router.post("/add-store")
async def add_store(store: StoreInput):
    store_data = {
        "name": store.name,
        "category": store.category,
        "latitude": store.latitude,
        "longitude": store.longitude,
    }

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
    model.stores.append([store_data["category"], geo, store_data["name"]])
    #TODO: investigate if we could get messed up by the id if a store gets deleted and now the ids are the same
    model.stores_list.append(Store(model=model, id=len(model.stores) + 1, name=name, type=store_data["category"], geometry=geo))
    #TODO: need type checking like category = SPM and name not in store_list
    return {"store_json": model.stores}

@router.get("/get-step-number")
async def get_step_number():
    step_number = model.schedule.steps
    return {"step_number": step_number}

@router.put("/step")
async def step():
    model.step()
    step_number = model.schedule.steps
    return {"step_number": step_number}

@router.get("/get-num-households")
async def get_num_households():
    num_households = len(model.households)
    return {"num_households": num_households}

@router.get("/get-num-stores")
async def get_num_stores():
    num_stores = len(model.stores)
    stores = model.stores_list
    numSPM = 0
    numNonSPM = 0
    for store in stores:
        if store is None:
            continue
        elif (
            store.type == "supermarket"
            or store.type == "greengrocer"
            or store.type == "grocery"
        ):
            numSPM += 1
        else:
            numNonSPM += 1
    return {"num_stores": num_stores, "numSPM": numSPM, "numNonSPM": numNonSPM}

@router.get("/get-household-stats")
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
 """