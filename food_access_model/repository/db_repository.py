import psycopg2
import asyncio
import time
import os
from typing import Dict, List, Any, Optional
from food_access_model.abm.geo_model import GeoModel

PASS = os.getenv("PASS")
APIKEY = os.getenv("APIKEY")
USER = os.getenv("USER")
NAME = os.getenv("NAME")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")


class DBRepository:
    """Singleton repository for database access and caching."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBRepository, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.households = None
            self.food_stores = None
            self.model = None

            # Database connection parameters

    async def initialize(self):
        """Initialize the repository by fetching data from database and creating a GeoModel instance to keep track of the model's structure and state"""
        if self.households is not None and self.food_stores is not None:
            return

        start_time = time.time()

        # Create connection pool
        connection = psycopg2.connect(
            host=HOST, database=NAME, user=USER, password=PASS, port=PORT
        )

        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM food_stores;")
            stores = cursor.fetchall()
            self.food_stores = stores
            # print(f"Fetched {len(stores)} food stores", flush=True)
            cursor.execute("SELECT * FROM households;")
            households = cursor.fetchall()
            self.households = households
            # print(f"Fetched {len(households)} households", flush=True)

        end_time = time.time()
        startup_duration = end_time - start_time
        print(f"Repository initialized in {startup_duration:.4f} seconds", flush=True)

        self.model = GeoModel(households=self.households, stores=self.food_stores)

        # async with connections_pool.acquire() as conn1, connections_pool.acquire() as conn2:
        #     try:
        #         # Execute queries in parallel
        #         query1 = conn1.fetch("SELECT * FROM food_stores;")
        #         query2 = conn2.fetch("SELECT * FROM households;")

        #         # Await both queries
        #         stores, households = await asyncio.gather(query1, query2)

        #         # Store results
        #         self.food_stores = stores
        #         self.households = households

        #         # Initialize model
        #         self.model = GeoModel(households=self.households, stores=self.food_stores)

        #         end_time = time.time()
        #         startup_duration = end_time - start_time
        #         print(f"Repository initialized in {startup_duration:.4f} seconds", flush=True)
        #     except Exception as e:
        #         print(f"Error initializing repository: {e}", flush=True)

    def get_model(self) -> GeoModel:
        """Get the GeoModel instance."""
        return self.model

    def update_model(self, new_households, new_stores, new_step: int = -1) -> None:
        """Update the model with new data."""
        updated_households = (
            new_households if new_households is not None else self.households
        )
        updated_stores = new_stores if new_stores is not None else self.food_stores

        updated_step = new_step if new_step != -1 else self.model.raw_step_number + 1

        self.model = GeoModel(households=updated_households, stores=updated_stores)
        self.model.set_step_number(updated_step)

    def get_households(self) -> List[Any]:
        """Get the households data."""
        return self.households

    def get_food_stores(self) -> List[Any]:
        """Get the food stores data."""
        return self.food_stores



    def is_initialized(self) -> bool:
        """Check if repository is initialized."""
        return self.households is not None and self.food_stores is not None


async def get_db_repository():
    repo = DBRepository()
    if not repo.is_initialized():
        print("Initializing repository")
        await repo.initialize()
    return repo
