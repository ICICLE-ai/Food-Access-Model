import os
import json
import logging
import asyncio
import uuid
from datetime import datetime

import asyncpg
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
import time

# SQLAlchemy’s async API
# DATABASE_URL = (
#     f"postgresql+asyncpg://{os.getenv('DB_USER')}:"
#     f"{os.getenv('DB_PASS')}@"
#     f"{os.getenv('DB_HOST')}:"
#     f"{os.getenv('DB_PORT')}/"
#     f"{os.getenv('DB_NAME')}"
# )

# pure asyncpg connection string
DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASS')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)


HOUSEHOLD_QUERY = """
                     SELECT
                     id,
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
                     WHERE simulation_instance = $1
                     AND simulation_step = $2;
                     """

FOOD_STORE_QUERY = """
                     SELECT
                     store_id,
                     shop,
                     geometry,
                     name
                     FROM food_stores
                     WHERE simulation_instance = $1
                     AND simulation_step = $2;
                     """

# database = databases.Database(DATABASE_URL)

router = APIRouter(prefix="/api", tags=["ABM"])
# FRONT_URL = os.environ.get("FRONT_URL", "http://localhost:5173")

pool: asyncpg.Pool = None  # will be set on startup


@router.on_event("startup")
async def startup():
    global pool
    logging.info(f"Initializing database connection to database {os.getenv('DB_NAME')} at {os.getenv('DB_HOST')}")
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    logging.info("Database pool created")


@router.on_event("shutdown")
async def shutdown():
    await pool.close()
    logging.info("Database pool closed")


@router.get("/simulation-instances")
async def get_simulation_instances() -> ORJSONResponse:
    """
    Get all simulation instances.

    Returns:
        dict: A dictionary containing a list of simulation instances.
    """
    query = """
        SELECT id::text AS id, name, description, created_at
        FROM simulation_instances
        ORDER BY created_at DESC;
        """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    simulation_instances = [dict(row) for row in rows]
    return ORJSONResponse({"simulation_instances": simulation_instances})


@router.get("/simulation-instances/{instance_id}")
async def get_simulation_instance(instance_id: str) -> ORJSONResponse:
    """
    Get a specific simulation instance by ID.

    Parameters:
        instance_id (str): The ID of the simulation instance to retrieve.

    Returns:
        dict: A dictionary containing the details of the simulation instance.
    """

    # check if instance_id is a valid uuid
    try:
        uuid.UUID(instance_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid instance ID format")

    query = """
        SELECT id::text AS id, name, description, created_at
        FROM simulation_instances
        WHERE id = $1;
        """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, instance_id)
    except Exception as e:
        logging.error(f"Error fetching simulation instance {instance_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

    if row is None:
        raise HTTPException(status_code=404, detail="Simulation instance not found")
    instance = dict(row)
    return ORJSONResponse({"simulation_instance": instance})


@router.post("/simulation-instances/{instance_id}/advance")
async def advance_simulation_instance(instance_id: str) -> ORJSONResponse:
    """
    Advance the simulation instance by one step.

    Parameters:
        instance_id (str): The ID of the simulation instance to advance

    Returns:
        dict: A dictionary indicating success
    """
    try:
        uuid.UUID(instance_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid instance ID format")

    await _run_model_step(instance_id)
    return ORJSONResponse({"status": "success"})


@router.post("/simulation-instances/{instance_id}/reset")
async def reset_simulation_instance(instance_id: str) -> ORJSONResponse:
    """
    Reset the simulation instance to its initial state and deletes data from other steps.

    Parameters:
        instance_id (str): The ID of the simulation instance to reset

    Returns:
        dict: A dictionary indicating success
    """
    await reset_simulation(instance_id)

    return {"result": f"Simulation instance {instance_id} reset successfully"}


@router.post("/simulation-instances")
async def create_simulation_instance(name: Optional[str] = Body(None, embed=True),
                                     description: Optional[str] = Body(None, embed=True)) -> ORJSONResponse:
    """
    Create a new simulation instance.

    Parameters:
        name (str, optional): The name of the simulation instance.
        description (str, optional): A description of the simulation instance.

    Returns:
        dict: A dictionary containing the details of the crceated simulation instance.
    """
    # Generate a name if not provided
    if name is None:
        name = generate_name()
    query = """
        INSERT INTO simulation_instances (name, description)
        VALUES ($1, $2)
        RETURNING id, name, description, created_at;
        """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, name, description)
    instance = dict(row)

    instance['id'] = str(instance['id'])  # Convert UUID to string for JSON serialization

    await generate_household_instances_for_simulation(instance['id'])
    await generate_stores_for_simulation(instance['id'])

    return ORJSONResponse({"simulation_instance": instance})


@router.delete("/simulation-instances/{instance_id}")
async def delete_simulation_instance(instance_id: str) -> ORJSONResponse:
    """
    Delete a simulation instance by ID.

    Parameters:
        instance_id (str): The ID of the simulation instance to delete.

    Returns:
        dict: A dictionary indicating success.
    """

    get_default_instance_id_query = """SELECT id FROM simulation_instances WHERE name = 'default_simulation';"""

    household_query = """
        DELETE FROM households
        WHERE simulation_instance = $1;
        """

    store_query = """
        DELETE FROM food_stores
        WHERE simulation_instance = $1;
        """

    instance_query = """
        DELETE FROM simulation_instances
        WHERE id = $1;
        """

    async with pool.acquire() as conn:
        default_instance_row = await conn.fetchrow(get_default_instance_id_query)

        if default_instance_row is None:
            raise HTTPException(status_code=404, detail="Default simulation instance not found")

        if instance_id == str(default_instance_row['id']):
            raise HTTPException(status_code=400, detail="Cannot delete the default simulation instance")

        await conn.execute(household_query, instance_id)
        await conn.execute(store_query, instance_id)
        await conn.execute(instance_query, instance_id)

    return ORJSONResponse({"status": "success", "message": "Simulation instance deleted"})


@router.get("/households")
async def get_all_households(simulation_instance: str = Query(..., description="Simulation instance ID"),
                             simulation_step: Optional[int] = Query(0, description="Optional step filter")) -> Dict[str, list]:
    """
    Gets all households in the model

    Parameters:
        simulation_instance (str): The ID of the simulation instance to get households for
        simulation_step (int): The step number to get households for

    Returns:
        dict: A dictionary of households in the model with 'households_json' as the key which has a list of
              household objects
    """
    household_data = await query_households(simulation_instance_id=simulation_instance, simulation_step=simulation_step)
    return ORJSONResponse({"households_json": household_data})


# Streaming endpoint for households - works here, but frontend needs to be updated to handle streaming
# @router.get("/households/stream")
# async def stream_json(step: Optional[int] = Query(0, description="Optional step filter")):
#     return StreamingResponse(stream_households(step), media_type="application/json")
#
# TODO: Update frontend to handle streaming
# TODO: adjust to use asyncpg for streaming
# async def stream_households(step: int):
#     """
#     Streams households data as a JSON array.
#     Parameters:
#         step (int): Optional step filter to limit the households returned.
#     Yields:
#         bytes: JSON-encoded households data.
#     """#

#     yield b"["
#     first = True
#     async with database.connection() as conn:
#         async for row in conn.iterate(HOUSEHOLD_QUERY, values={"step": step}):
#             item_json = orjson.dumps(dict(row))
#             if first:
#                 yield item_json
#                 first = False
#             else:
#                 yield b"," + item_json
#             await asyncio.sleep(0)
#     yield b"]"


@router.get("/stores")
async def get_stores(simulation_instance: str = Query(..., description="Simulation instance ID"),
                     simulation_step: Optional[int] = Query(0, description="Optional step filter")) -> Dict[str, list]:
    """
    Gets all stores in the model

    Parameters:
        simulation_instance (str): The ID of the simulation instance to get stores for
        simulation_step (int): The step number to get stores for

    Returns:
        dict: A dictionary of stores in the model with 'store_json' as the key which has a list of store objects
    """
    food_stores = await query_food_stores(simulation_instance_id=simulation_instance, simulation_step=simulation_step)
    return {"store_json": food_stores}


@router.post("/stores")
async def add_store(store: StoreInput) -> Dict[str, List[Dict[str, Any]]]:
    """
    Adds a store to the model

    Parameters:
        store (StoreInput): The details of the store to be added.
                            Dict should contain 'name', 'category', 'latitude', 'longitude', 'simulation_instance_id', and 'simulation_step'.

    Returns:
        dict: A dictionary of stores in the simulation with the new added store

    """
    # convert latitude and longitude to a polygon
    geo = str(
        convert_centroid_to_polygon(
            store.latitude, store.longitude, store.category
        )
    )

    # get the highest store_id for the simulation instance and step
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT MAX(store_id) AS max_id
            FROM food_stores
            WHERE simulation_instance = $1 AND simulation_step = $2
        """, store.simulation_instance_id, store.simulation_step)
        max_id = row['max_id'] if row and row['max_id'] is not None else 0
        new_store_id = max_id + 1

    async with pool.acquire() as conn:
        # Insert the new store
        await conn.fetchrow("""
            INSERT INTO food_stores (name, shop, geometry, simulation_instance, simulation_step, store_id)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, store.name, store.category, geo, store.simulation_instance_id, store.simulation_step, new_store_id)

        # get all of the stores for the simulation instance and timestep
        stores = await query_food_stores(simulation_instance_id=store.simulation_instance_id,
                                         simulation_step=store.simulation_step)

        return {"store_json": stores}


@router.delete("/stores")
async def remove_store(store_id: str = Query(..., description="ID of the store to delete"),
                       simulation_instance_id: str = Query(..., description="Simulation instance ID"),
                       simulation_step: int = Query(..., description="Simulation step")) -> Dict[str, List[Dict[str, Any]]]:
    """
    Removes a store from a model given a store_id, simulation_instance_id, and simulation_step

    Parameters:
        store_id (str): The ID of the store to be removed
        simulation_instance_id (str): The ID of the simulation instance to remove the store from
        simulation_step (int): The step number to remove the store from

    Returns:
        dict: A dictionary of the updated stores in the model as well as the removed store
    """
    async with pool.acquire() as conn:
        # Find the store to remove
        store = await conn.fetchrow("""
            SELECT * FROM food_stores
            WHERE store_id = $1 AND simulation_instance = $2 AND simulation_step = $3
        """, int(store_id), simulation_instance_id, simulation_step)
        if store is None:
            raise HTTPException(status_code=404, detail="Store not found")

        # Delete the store
        await conn.execute("""
            DELETE FROM food_stores
            WHERE store_id = $1 AND simulation_instance = $2 AND simulation_step = $3
        """, int(store_id), simulation_instance_id, simulation_step)

        # Get all stores after deletion
        stores = await query_food_stores(simulation_instance_id=simulation_instance_id,
                                         simulation_step=simulation_step)
        return {"store_json": stores}


@router.get("/get-step-number")
async def get_step_number(simulation_instance: str = Query(..., description="Simulation instance ID")) -> Dict[str, int]:
    """
    Gets the current step number the model is at

    Parameters:
        simulation_instance (str): The ID of the simulation instance to get the step number for

    Returns:
        dict: A dictionary with the current step number
    """
    step_number = await query_current_simulation_step(simulation_instance)
    return {"step_number": step_number}


@router.get("/get-num-households")
async def get_num_households(simulation_instance_id: str = Query(..., description="Simulation instance ID"),
                             simulation_step: int = Query(..., description="Simulation step")) -> Dict[str, int]:
    """
    Gets the number of households in the model

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to get the number of households for
        simulation_step (int): The step number to get the number of households for

    Returns:
        dict: Dictionary with the number of households in the model
    """
    async with pool.acquire() as conn:
        # Find the store to remove
        row = await conn.fetchrow("""
            SELECT count(*) FROM households
            WHERE simulation_instance = $1 AND simulation_step = $2
            """, simulation_instance_id, simulation_step)

    household_count = row['count'] if row else 0

    return {"num_households": household_count}


@router.get("/get-num-stores")
async def get_num_stores(simulation_instance_id: str = Query(..., description="Simulation instance ID"),
                         simulation_step: int = Query(..., description="Simulation step")) -> Dict[str, int]:
    """
    Gets the number of stores in the model

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to get the number of stores for
        simulation_step (int): The step number to get the number of stores for

    Returns:
        dict: Dictionaries with the number of stores, number of supermarket stores, and number of non-supermarket stores
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                CASE
                    WHEN shop IN ('supermarket', 'greengrocer', 'grocery') THEN 'numSPM'
                    ELSE 'numNonSPM'
                END AS store_group,
                COUNT(*) AS store_count
            FROM food_stores
            WHERE simulation_instance = $1 AND simulation_step = $2
            GROUP BY store_group
            ORDER BY store_group
            """, simulation_instance_id, simulation_step)

    store_counts = {row['store_group']: row['store_count'] for row in rows}

    return {'num_stores': sum(store_counts.values()),
            'numSPM': store_counts.get('numSPM', 0),
            'numNonSPM': store_counts.get('numNonSPM', 0)}


@router.get("/get-household-stats")
async def get_household_stats(simulation_instance_id: str = Query(..., description="Simulation instance ID"),
                              simulation_step: int = Query(..., description="Simulation step")) -> Dict[str, float]:
    """
    Gets household stats

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to get the household stats for
        simulation_step (int): The step number to get the household stats for

    Returns:
        dict: Dictionaries with average income of the households and average vehicles per household
    """
    households = await query_households(simulation_instance_id, simulation_step)
    income = 0
    vehicles = 0
    for house in households:
        income += house['Income']
        vehicles += house['Vehicles']
    avg_income = float(income) / len(households)
    avg_vehicles = float(vehicles) / len(households)
    return {"avg_income": avg_income, "avg_vehicles": avg_vehicles}


@router.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running properly"""
    try:
        # Basic check that we can access the database
        async with pool.acquire() as conn:
            await conn.fetch("SELECT 1")  # Simple query to check database connection
        return {"status": "healthy", "message": "API is operational"}
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


async def query_current_simulation_step(simulation_instance_id: str) -> int:
    """
    Queries the current simulation step for a specific simulation instance.

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to query.

    Returns:
        int: The current step number.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT MAX(simulation_step) AS current_step FROM households WHERE simulation_instance = $1",
            simulation_instance_id
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
    async with pool.acquire() as conn:
        rows = await conn.fetch(HOUSEHOLD_QUERY, simulation_instance_id, simulation_step)

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
    async with pool.acquire() as conn:
        rows = await conn.fetch(FOOD_STORE_QUERY, simulation_instance_id, simulation_step)

    # Convert rows to a list of dictionaries
    food_stores_data = [dict(row) for row in rows]
    return food_stores_data


async def _run_model_step(simulation_instance_id) -> None:
    """
    Runs one step of the model

    Parameters:
        simulation_instance_id (str): The ID of the simulation instance to run the step for
    """
    start_time = time.time()
    current_step = await query_current_simulation_step(simulation_instance_id)
    logging.info(f"Step {current_step}: Queried current simulation step in {time.time() - start_time:.3f}s")

    t1 = time.time()
    households = await query_households(simulation_instance_id=simulation_instance_id, simulation_step=current_step)
    logging.info(f"Step {current_step}: Queried {len(households)} households in {time.time() - t1:.3f}s")

    t2 = time.time()
    food_stores = await query_food_stores(simulation_instance_id=simulation_instance_id, simulation_step=current_step)
    logging.info(f"Step {current_step}: Queried {len(food_stores)} food stores in {time.time() - t2:.3f}s")

    t3 = time.time()
    batch_results = await batch_run_model(households=households, food_stores=food_stores)
    logging.info(f"Step {current_step}: Ran model step in {time.time() - t3:.3f}s")

    t4 = time.time()
    await return_step_results_to_database(households=batch_results['households'],
                                          simulation_instance_id=simulation_instance_id,
                                          simulation_step=current_step + 1)

    logging.info(f"Step {current_step}: Saved step results to database in {time.time() - t4:.3f}s")

    logging.info(f"Step {current_step}: Total time for _run_model_step: {time.time() - start_time:.3f}s")


async def reset_simulation(instance_id: str) -> None:
    """
    Resets the simulation instance by deleting all households and food stores.

    Parameters:
        instance_id (str): The ID of the simulation instance to reset.
    """
    # TODO: Doesn't delete any stores that were added in step 0, only those added in later steps
    # The simulation should probably be run once on instantiation in order to properly initialize households
    async with pool.acquire() as conn:
        # Delete all households for the given simulation instance
        await conn.execute(
            "DELETE FROM households WHERE simulation_instance = $1 and simulation_step != 0", instance_id
        )
        # Delete all food stores for the given simulation instance
        await conn.execute(
            "DELETE FROM food_stores WHERE simulation_instance = $1 and simulation_step != 0", instance_id
        )


async def batch_run_model(households: List[Dict[str, Any]], food_stores: List[Dict[str, Any]]) -> None:
    """
    Runs the model in batch mode using the provided households and food stores.
    Parameters:
        households (List[Dict[str, Any]]): A list of household data dictionaries
        food_stores (List[Dict[str, Any]]): A list of food store data dictionaries
    """
    results = batch_run(
        GeoModel,
        parameters={
            "households": households,
            "stores": food_stores,
        },
        max_steps=1,
        data_collection_period=1,
        display_progress=True,
        number_processes=25,
    )
    # at this point, stores do not have an id
    all_households = []
    finalResults = filter_unique_by_runid(results)

    for res in results:
        if isinstance(res, dict):
            all_households.extend(res.get("households", []))

        elif isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    all_households.extend(item.get("households", []))

    all_stores = finalResults[0].get("stores", [])

    return {"households": all_households, "stores": all_stores}


def filter_unique_by_runid(results) -> Dict[str, Any]:
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


async def return_step_results_to_database(households: List[Dict[str, Any]],
                                          simulation_instance_id: str,
                                          simulation_step: int) -> None:
    """
    Updates the agents in the database for a specific simulation instance and step.
    """

    async with pool.acquire() as conn:
        # Delete existing records for the current step
        await conn.execute(
            "DELETE FROM households WHERE simulation_instance = $1 AND simulation_step = $2",
            simulation_instance_id,
            simulation_step
        )
        await conn.execute(
            "DELETE FROM food_stores WHERE simulation_instance = $1 AND simulation_step = $2",
            simulation_instance_id,
            simulation_step
        )

        # Insert new records for households
        if households:
            t_insert = time.time()
            try:
                await conn.copy_records_to_table(
                    'households',
                    records=(
                        (
                            house["id"],
                            simulation_instance_id,
                            simulation_step,
                            house["Geometry"],
                            house["Income"],
                            house["Household Size"],
                            house["Vehicles"],
                            house["Number of Workers"],
                            house["Transit time"],
                            house["Walking time"],
                            house["Biking time"],
                            house["Driving time"]
                        )
                        for house in households  # generator: no big list in memory
                    ),
                    columns=[
                        "id",
                        "simulation_instance",
                        "simulation_step",
                        "centroid_wkt",
                        "income",
                        "household_size",
                        "vehicles",
                        "number_of_workers",
                        "transit_time",
                        "walking_time",
                        "biking_time",
                        "driving_time"
                    ]
                )
                logging.info(f"Inserted {len(households)} households via COPY in {time.time() - t_insert:.3f}s")
            except Exception as e:
                logging.error(f"Error inserting households via COPY: {e}")

        # Insert new records for food stores
        # just need to copy the stores from the previous step, as simulation doesn't change stores
        async with pool.acquire() as conn:
            insert_query = """
                INSERT INTO food_stores (
                    simulation_instance, simulation_step, name, shop, geometry, store_id
                )
                SELECT $1, $2, name, shop, geometry, store_id
                FROM food_stores
                WHERE simulation_instance = $1 AND simulation_step = $3;
            """

            await conn.execute(insert_query, simulation_instance_id, simulation_step, simulation_step - 1)


async def generate_household_instances_for_simulation(instance_id: str) -> None:
    """
    Generates household instances for a given simulation instance.

    Parameters:
        instance_id (str): The ID of the simulation instance to generate households for.
    """
    # Placeholder for actual implementation
    # This function should create household instances in the database for the given simulation instance
    # TODO: for now we'll just copy the households from the default simulation instance and step = 0
    query = """
        INSERT INTO households (
            simulation_instance, simulation_step, id, centroid_wkt, income, household_size,
            vehicles, number_of_workers, transit_time, walking_time, biking_time, driving_time
        )
        SELECT $1, 0, id, centroid_wkt, income, household_size, vehicles, number_of_workers,
            transit_time, walking_time, biking_time, driving_time
        FROM households
        WHERE simulation_instance = $2 AND simulation_step = 0;
    """

    # Assuming you already have an asyncpg connection object
    get_default_instance_id_query = """SELECT id FROM simulation_instances WHERE name = 'default_simulation';"""

    async with pool.acquire() as conn:
        default_instance_row = await conn.fetchrow(get_default_instance_id_query)
        if default_instance_row is None:
            raise HTTPException(status_code=404, detail="Default simulation instance not found")

        default_instance_id = default_instance_row["id"]

        await conn.execute(query, instance_id, default_instance_id)


async def generate_stores_for_simulation(instance_id: str):
    """
    Generates store instances for a given simulation instance.

    Parameters:
        instance_id (str): The ID of the simulation instance to generate stores for.
    """
    # Placeholder for actual implementation - currently just copies the stores from the
    # default simulation instance and step = 0
    # This function should create store instances in the database for the given simulation instance

    # Get the simulation_instance id for the default simulation
    get_default_instance_id_query = """SELECT id FROM simulation_instances WHERE name = 'default_simulation';"""

    async with pool.acquire() as conn:
        default_instance_row = await conn.fetchrow(get_default_instance_id_query)
        if default_instance_row is None:
            raise HTTPException(status_code=404, detail="Default simulation instance not found")

        default_instance_id = default_instance_row["id"]

        insert_query = """
            INSERT INTO food_stores (
                simulation_instance, simulation_step, name, shop, geometry, store_id
            )
            SELECT $1, 0, name, shop, geometry, store_id
            FROM food_stores
            WHERE simulation_instance = $2 AND simulation_step = 0;
        """

        await conn.execute(insert_query, instance_id, default_instance_id)


async def generate_stores_for_simulation_step(instance_id: str, simulation_step: int) -> None:
    """
    Generates store instances for a given simulation instance.

    Parameters:
        instance_id (str): The ID of the simulation instance to generate stores for.
        simulation_step (int): The step number to generate stores for.
    """

    async with pool.acquire() as conn:

        insert_query = """
            INSERT INTO food_stores (
                simulation_instance, simulation_step, name, shop, geometry, store_id
            )
            SELECT $1, $2, name, shop, geometry, store_id
            FROM food_stores
            WHERE simulation_instance = $1 AND simulation_step = $3;
        """

        await conn.execute(insert_query, instance_id, simulation_step, simulation_step - 1)


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
    # # If the store name exists in either list, return an error message
    # (should change to id later on but doing this for MVP)
    if store_exists_in_stores or store_exists_in_stores_list:
        raise HTTPException(status_code=409, detail=f"Store with name '{name}' already exists.")

    #convert latitude and longitude to a polygon
    geo = str(convert_centroid_to_polygon(store_data["latitude"], store_data["longitude"], store_data["category"]))
    #does id matter if some get deleted, like does some operation rely on them being contiguous?
    model.stores.append([store_data["category"], geo, store_data["name"]])
    #TODO: investigate if we could get messed up by the id if a store gets deleted and now the ids are the same
    model.stores_list.append(Store(model=model,
                                   id=len(model.stores) + 1,
                                   name=name,
                                   type=store_data["category"],
                                   geometry=geo))
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
