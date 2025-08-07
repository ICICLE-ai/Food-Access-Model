import os
import json
import logging
import asyncio
import uuid
from datetime import datetime

from typing import List, Dict, Union, Any, Optional
from names_generator import generate_name
import databases
import orjson

from fastapi import APIRouter, Body, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse, ORJSONResponse

from food_access_model.api.helpers import StoreInput, convert_centroid_to_polygon
from food_access_model.abm.geo_model import GeoModel
from food_access_model.abm.store import Store
from food_access_model.repository.db_repository import DBRepository, get_db_repository
from food_access_model.model_multi_processing.batch_running import batch_run


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
                     WHERE simulation_instance = :instance
                     AND step = :step;
                     """

FOOD_STORE_QUERY = """
                     SELECT
                     shop,
                     geometry,
                     name
                     FROM food_stores
                     WHERE simulation_instance = :instance
                     AND simulation_step = :step;
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


@router.get("/simulation-instances")
async def get_simulation_instances():
    query = "SELECT id::text AS id, name, description, created_at FROM simulation_instances ORDER BY created_at DESC;"
    async with database.connection() as conn:
        rows = await conn.fetch_all(query)
    simulation_instances = [dict(row) for row in rows]
    return ORJSONResponse({"simulation_instances": simulation_instances})


@router.get("/simulation-instances/{instance_id}")
async def get_simulation_instance(instance_id: str):
    query = "SELECT id::text AS id, name, description, created_at FROM simulation_instances WHERE id = :id;"
    values = {"id": instance_id}
    async with database.connection() as conn:
        row = await conn.fetch_one(query, values)
    if row is None:
        raise HTTPException(status_code=404, detail="Simulation instance not found")
    instance = dict(row)
    return ORJSONResponse({"simulation_instance": instance})


@router.post("/simulation-instances")
async def create_simulation_instance(
    name: Optional[str] = Body(None, embed=True),
    description: Optional[str] = Body(None, embed=True)
):
    # Generate a name if not provided
    if not name:
        name = generate_name()
    query = "INSERT INTO simulation_instances (name, description) VALUES (:name, :description) RETURNING id, name, description, created_at;"
    values = {"name": name, "description": description}

    async with database.connection() as conn:
        row = await conn.fetch_one(query, values)
    instance = dict(row)

    generate_household_instances_for_simulation(instance['id'])
    generate_stores_for_simulation(instance['id'])

    return ORJSONResponse({"simulation_instance": instance})


# @router.get("/stores")
# async def get_stores(repository: DBRepository = Depends(get_db_repository))-> Dict[str, list]:
#     """
#     Gets all stores from the model
# 
#     Parameters:
#         repository (DBRepository): A singleton interface to the simulation model that gives access to the households, stores,
#         and other data required to initialize the simulation
# 
#     Returns:
#         dict: A dictionary with a key 'stores_json' which has a list of unspecified type
#     """
#     model = repository.get_model()
#     stores = model.get_stores()
#     return {"stores_json": stores}


@router.get("/stores")
async def get_stores(simulation_instance: str = Query(..., description="Simulation instance ID"), simulation_step: Optional[int] = Query(0, description="Optional step filter")) -> Dict[str, list]:
    """
    Gets all stores from the model

    Parameters:
        simulation_instance (str): The ID of the simulation instance
        simulation_step (Optional[int]): The simulation step to filter stores

    Returns:
        dict: A dictionary with a key 'stores_json' which has a list of stores
    """
    stores_data = await query_food_stores(simulation_instance_id=simulation_instance, simulation_step=simulation_step)
    return ORJSONResponse({"stores_json": stores_data})


@router.get("/households")
async def get_all_households(simulation_instance: str = Query(..., description="Simulation instance ID"), simulation_step: Optional[int] = Query(0, description="Optional step filter")) -> Dict[str, list]:
    household_data = await query_households(simulation_instance_id=simulation_instance, simulation_step=simulation_step)
    return ORJSONResponse({"households_json": household_data})


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
    async with database.connection() as conn:
        async for row in conn.iterate(HOUSEHOLD_QUERY, values={"step": step}):
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
    households = model.get_households().astype(str)
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


async def _run_model_step(simulation_instance_id)->None:
    """
    Runs one step of the model

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to run the step for
    """
    current_step = await query_current_simulation_step(simulation_instance_id)
    households = await query_households(simulation_instance_id=simulation_instance_id, simulation_step=current_step)
    food_stores = await query_food_stores(simulation_instance_id=simulation_instance_id, simulation_step=current_step)
    model = GeoModel(households=households, stores=food_stores)
    model.step()
    return_step_results_to_database(model=model, simulation_instance_id=simulation_instance_id, step=current_step + 1)


async def query_current_simulation_step(simulation_instance_id: str) -> int:
    """
    Queries the current simulation step for a specific simulation instance.

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to query.

    Returns:
        int: The current step number.
    """
    async with database.connection() as conn:
        row = await conn.fetch_one(
            "SELECT MAX(step) AS current_step FROM households WHERE simulation_instance = :instance",
            values={"instance": simulation_instance_id}
        )
    if row is None or row["current_step"] is None:
        return 0
    return row["current_step"]


async def query_households(simulation_instance_id: str, simulation_step: int = 0) -> List[Any]:
    """
    Queries the households for a specific simulation instance and step.

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to query.
        simulation_step (int, optional): The simulation step to query. Defaults to 0.

    Returns:
        List[Any]: A list of household data.
    """
    async with database.connection() as conn:
        rows = await conn.fetch_all(HOUSEHOLD_QUERY, values={"instance": simulation_instance_id, "step": simulation_step})

    # Convert rows to a list of dictionaries
    households_data = [dict(row) for row in rows]
    return households_data


async def query_food_stores(simulation_instance_id: str, simulation_step: int = 0) -> List[Any]:
    """
    Queries the food stores for a specific simulation instance and step.

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to query.
        simulation_step (int, optional): The simulation step to query. Defaults to 0.

    Returns:
        List[Any]: A list of food store data.
    """
    async with database.connection() as conn:
        rows = await conn.fetch_all(FOOD_STORE_QUERY, values={"instance": simulation_instance_id, "step": simulation_step})

    # Convert rows to a list of dictionaries
    food_stores_data = [dict(row) for row in rows]
    return food_stores_data


async def return_step_results_to_database(model: GeoModel, simulation_instance_id: str, simulation_step: int) -> None:
    """
    Updates the agents in the database for a specific simulation instance and step.
    """
    stores = model.get_food_stores()
    households = model.get_households()

    async with database.connection() as conn:
        # Delete existing records for the current step
        await conn.execute(
            "DELETE FROM households WHERE simulation_instance = :instance AND step = :step",
            values={"instance": simulation_instance_id, "step": simulation_step}
        )
        await conn.execute(
            "DELETE FROM stores WHERE simulation_instance = :instance AND step = :step",
            values={"instance": simulation_instance_id, "step": simulation_step}
        )

        # Insert new records for households
        if households:
            await conn.execute_many(
                "INSERT INTO households (simulation_instance, step, centroid_wkt, income, household_size, vehicles, number_of_workers, transit_time, walking_time, biking_time, driving_time) VALUES (:instance, :step, :centroid_wkt, :income, :household_size, :vehicles, :number_of_workers, :transit_time, :walking_time, :biking_time, :driving_time)",
                [
                    {
                        "instance": simulation_instance_id,
                        "step": step,
                        "centroid_wkt": house["Geometry"],
                        "income": house["Income"],
                        "household_size": house["Household Size"],
                        "vehicles": house["Vehicles"],
                        "number_of_workers": house["Number of Workers"],
                        "transit_time": house["Transit time"],
                        "walking_time": house["Walking time"],
                        "biking_time": house["Biking time"],
                        "driving_time": house["Driving time"]
                    } for house in households
                ]
            )

        # Insert new records for food stores
        if stores:
            await conn.execute_many(
                "INSERT INTO stores (simulation_instance, step, name, category, latitude, longitude) VALUES (:instance, :step, :name, :category, ST_Y(ST_GeomFromText(:geometry)), ST_X(ST_GeomFromText(:geometry)))",
                [
                    {
                        "instance": simulation_instance_id,
                        "step": simulation_step,
                        "name": store["Name"],
                        "category": store["Category"],
                        "geometry": store["Geometry"]
                    } for store in stores
                ]
            )


@router.put("/step")
async def step(simulation_instance_id: str):
    """
    Runs one step of the model

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to run the step for

    Returns:
        None
    """

    await _run_model_step(simulation_instance_id)


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


async def generate_household_instances_for_simulation(instance_id: str):
    """
    Generates household instances for a given simulation instance.

    Parameters:
        instance_id (str): The ID of the simulation instance to generate households for.
    """
    # Placeholder for actual implementation
    # This function should create household instances in the database for the given simulation instance
    # TODO: for now we'll just copy the households from the default simulation instance and step = 0
    query = """
    INSERT INTO households (simulation_instance, step, centroid_wkt, income, household_size, vehicles, number_of_workers, transit_time, walking_time, biking_time, driving_time)
    SELECT :instance_id, 0, centroid_wkt, income, household_size, vehicles, number_of_workers, transit_time, walking_time, biking_time, driving_time
    FROM households
    WHERE simulation_instance = 'default' AND step = 0;
    """
    values = {"instance_id": instance_id}
    async with database.connection() as conn:
        await conn.execute(query, values)


async def generate_stores_for_simulation(instance_id: str):
    """
    Generates store instances for a given simulation instance.

    Parameters:
        instance_id (str): The ID of the simulation instance to generate stores for.
    """
    # Placeholder for actual implementation
    # This function should create store instances in the database for the given simulation instance
    # Get the simulation_instance id for the default simulation
    get_default_instance_id_query = "SELECT id FROM simulation_instances WHERE name = 'default_simulation';"
    async with database.connection() as conn:
        default_instance_row = await conn.fetch_one(get_default_instance_id_query)
        if default_instance_row is None:
            raise HTTPException(status_code=404, detail="Default simulation instance not found")
        default_instance_id = default_instance_row["id"]

    query = """
    INSERT INTO stores (simulation_instance, step, name, category, latitude, longitude)
    SELECT :instance_id, 0, name, category, latitude, longitude
    FROM stores
    WHERE simulation_instance = :default_instance_id AND step = 0;
    """
    values = {"instance_id": instance_id, "default_instance_id": default_instance_id}
    values = {"instance_id": instance_id}
    async with database.connection() as conn:
        await conn.execute(query, values)


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
