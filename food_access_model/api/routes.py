import logging
import json
import os

from fastapi.responses import StreamingResponse
import orjson
import asyncio
import databases

from fastapi import APIRouter, Body, HTTPException, Depends, Query
from food_access_model.api.helpers import StoreInput, convert_centroid_to_polygon
from food_access_model.abm.geo_model import GeoModel
from food_access_model.model_multi_processing.batch_running import batch_run
from food_access_model.abm.store import Store
from food_access_model.repository.db_repository import DBRepository, get_db_repository
from food_access_model.abm.geo_model import GeoModel
from typing import List, Dict, Union, Any, Optional

DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

HOUSEHOLD_QUERY = """
                     SELECT 
                     'household' AS "Type",
                     centroid_wkt AS "Geometry",
                     income AS "Income",
                     household_size AS "Household Size",
                     vehicles AS "Vehicles",
                     number_of_workers AS "Number of Workers",
                     NULL AS "Stores within 1 Mile",
                     NULL AS "Closest Store (Miles)",
                     NULL AS "Rating for Distance to Closest Store",
                     NULL AS "Rating for Number of Stores within 1.0 Miles",
                     NULL AS "Ratings Based on Num of Vehicle",
                     transit_time AS "Transit time",
                     walking_time AS "Walking time",
                     biking_time AS "Biking time",
                     driving_time AS "Driving time",
                     NULL AS "Food Access Score",
                     NULL AS "Color"
                     FROM households
                     WHERE step = :step;
                     """

database = databases.Database(DATABASE_URL)

router = APIRouter(prefix="/api", tags=["ABM"])
#FRONT_URL = os.environ.get("FRONT_URL", "http://localhost:5173")

@router.on_event("startup")
async def startup():
    await database.connect()

@router.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@router.get("/stores")
async def get_stores(repository: DBRepository = Depends(get_db_repository))-> Dict[str, list]:
    """
    Gets all stores from the model

    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores,
        and other data required to initialize the simulation

    Returns:
        dict: A dictionary with a key 'stores_json' which has a list of unspecified type
    """
    model = repository.get_model()
    stores = model.get_stores()
    return {"stores_json": stores}


@router.get("/agents")
async def get_agents(repository: DBRepository = Depends(get_db_repository))->Dict[str, list]:
    """
    Gets all agents from the model

    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation

    Returns:
     dict: A dictionary of agents in the model with 'agents_json' as the key which has a list of unspecified type
    """
    model = repository.get_model()
    agents = model.agents
    return {"agents_json": agents}


# new and improved batched households endpoint
from fastapi.responses import ORJSONResponse

@router.get("/households")
async def get_all_households(step: Optional[int] = Query(0, description="Optional step filter")):
    query = "SELECT id, centroid_wkt as geom, income, household_size, vehicles, number_of_workers, walking_time, biking_time, transit_time, driving_time FROM households where step = :step"
    async with database.connection() as conn:
        rows = await conn.fetch_all(HOUSEHOLD_QUERY, values={"step": step})

    # Convert rows to a list of dictionaries
    households_data = [dict(row) for row in rows]

    return ORJSONResponse({"households_json": households_data})

# Streaming endpoint for households - works here, but frontend needs to be updated to handle streaming
async def stream_households(step: int):
    """
    Streams households data as a JSON array.
    Parameters:
        step (int): Optional step filter to limit the households returned.
    Yields:
        bytes: JSON-encoded households data.
    """

    yield b"["
    first = True
    query = "SELECT id, centroid_wkt as geom, income, household_size, vehicles, number_of_workers, walking_time, biking_time, transit_time, driving_time FROM households where step = :step"
    async with database.connection() as conn:
        async for row in conn.iterate(query, values={"step": step}):
            item_json = orjson.dumps(dict(row))
            if first:
                yield item_json
                first = False
            else:
                yield b"," + item_json
            await asyncio.sleep(0)
    yield b"]"


@router.get("/households/stream")
async def stream_json(step: Optional[int] = Query(0, description="Optional step filter")):
    return StreamingResponse(stream_households(step), media_type="application/json")


# Original route definition for households
@router.get("/households_old")
async def get_households(repository: DBRepository = Depends(get_db_repository)):#  -> Dict[str, list]:
    """
    Gets all households from the model

    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation

    Returns:
        dict: A dictionary of households in the model with 'households_json' as the key which has a list of unspecified type
    """
    model = repository.get_model()
    households = await model.get_households().astype(str)
    return {"households_json": households.to_dict(orient="records") }

    '''
    # return a subset of fields
    households['Type']="household"
    min_info = households[["Type", "Geometry", "Color"]].reset_index()
    households_json = min_info[['Type', 'AgentID', 'Geometry', 'Color']].to_dict(orient="records")
    return {"households_json": households_json}
    '''


@router.delete("/remove-store")
async def remove_store(store_name: str = Body(...), repository: DBRepository = Depends(get_db_repository))->Dict[str, list]:
    """
    Removes a store from a model given a name

    Parameters:
        store_name (str): The name of the store to be removed
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation

    Returns:
        dict: A dictionary of the updated stores in the model as well as the removed store
    """
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
async def reset_all(repository: DBRepository = Depends(get_db_repository))->Dict[str, list]:
    """
    Resets the stores in the model

    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation

    Returns:
        dict: An empty dictionary (of stores)
    """
    model = repository.get_model()
    # currently only resets stores, do we want to reset steps too?
    model.reset_stores()
    return {"store_json": model.stores}


@router.post("/add-store")
async def add_store(store: StoreInput, repository: DBRepository = Depends(get_db_repository))->Dict[str, list]:
    """
    Adds a store to the model

    Parameters:
        store (StoreInput): The store object being inputted consisting of name, category, latitude, and longitude
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation

    Returns:
        dict: A dictionary of stores in the model with the new added store
    """
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
async def get_step_number(repository: DBRepository = Depends(get_db_repository))->Dict[str, int]:
    """
    Gets the current step number the model is at
    
    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation
 
    Returns:
        dict: A dictionary with the current step number
    """
    model = repository.get_model()
    step_number = model.schedule.steps
    return {"step_number": step_number}

def filter_unique_by_runid(results)->Dict[str, Any]:
    """
    Filters a list/dictionary by id and returns a list/dictionary with unique ids
    
    Parameters:
        results (list): A dictionary of ids or a list of dictionaries with an id as a key
    
    Returns:
        list: A list of unique values to a dictionary with key 'runid'
    """
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

async def _run_model_step(repository: DBRepository)->int:
    """
    Runs one step of the model
        
    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation
        
    Returns:
        int: The current time step in the simulation
    """
    stores = await repository.get_food_stores()
    households = await repository.get_households()

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


    await repository.update_model(all_households, allStores)
    #print("Model update completed", flush=True)
    step_number = repository.get_model().raw_step_number
    return step_number

@router.put("/step")
async def step(repository: DBRepository = Depends(get_db_repository))->Dict[str, int]:
    """
    Runs one step of the model
        
    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation
        
    Returns:
        dict: Dictionary with the total number of steps taken
    """

    """ step_number = await _run_model_step(repository) """
    step_number =  await _run_model_step(repository)

    return {"step_number": step_number}


@router.get("/get-num-households")
async def get_num_households(repository: DBRepository = Depends(get_db_repository))->Dict[str, int]:
    """
    Gets the number of households in the model
        
    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation
        
    Returns:
        dict: Dictionary with the number of households in the model
"""
    model = repository.get_model()
    num_households = len(model.households)
    return {"num_households": num_households}


@router.get("/get-num-stores")
async def get_num_stores(repository: DBRepository = Depends(get_db_repository))->Dict[str, int]:
    """
    Gets the number of stores in the model
        
    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation
        
    Returns:
        dict: Dictionaries with the number of stores, number of supermarket stores, and number of non-supermarket stores
"""
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
async def get_household_stats(repository: DBRepository = Depends(get_db_repository))->Dict[str, float]:
    """
    Gets household stats
        
    Parameters:
        repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores, 
        and other data required to initialize the simulation
        
    Returns:
        dict: Dictionaries with average income of the households and average vehicles per household
"""
    
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


@router.get("/health")
async def health_check(repository: DBRepository = Depends(get_db_repository)):
    """Health check endpoint to verify the API is running properly"""
    try:
        # Basic check that we can access the model
        model = repository.get_model()
        return {"status": "healthy", "message": "API is operational"}
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")



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
