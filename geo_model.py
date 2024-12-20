from mesa import Model, DataCollector #Base class for GeoModel
from mesa.time import RandomActivation #Used to specify that agents are run randomly within each step
from mesa_geo import GeoSpace #GeoSpace that houses agents
import pandas as pd
from store import Store # Store agent class
from household import Household # Household agent class
import psycopg2
import os

DB_PASS = os.getenv("DB_PASS")
APIKEY = os.getenv("APIKEY")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

from constants import(
    SEARCHRADIUS,
    CRS
)

class GeoModel(Model):
    """
    Geographical model that extends the mesa Model base class. This class initializes the store and household agents
    and then places the agents in the mesa_geo GeoSpace, which allows the Household agents to calculate distances between
    between themselves and Store Agents.
    """

    def __init__(self):
        """
        Initialize the Model, intialize all agents and, add all agents to GeoSpace and Model.

        Args:
            - stores: dataframe containing data for store agents
            - households: dataframe containing data for household agents
        """
        super().__init__() 
        # Create new GeoSpace to contain agents
        self.space = GeoSpace(warn_crs_conversion=False) 
        # Specify that agents should be activated randomly during each step
        self.schedule = RandomActivation(self) 
        # Initializing empty list to collect all the store objects
        self.stores_list = []


        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        cursor = connection.cursor()

        # Execute the SQL query
        cursor.execute("SELECT * FROM food_stores;")

        # Fetch all rows from the executed query
        self.stores = cursor.fetchall()

        # Execute the SQL query
        cursor.execute("SELECT * FROM households;")

        # Fetch all rows from the executed query
        self.households = cursor.fetchall()

        cursor.close()
        connection.close()

        # Initialize all store agents and add them to the GeoSpace
        index_count = 0
        for store in self.stores:
            agent = Store(
                self, 
                index_count + len(self.households), 
                store[2], #name
                store[0], #shop
                store[1] #geo
                )
            index_count+=1
            self.space.add_agents(agent) 
            # Initializing empty list to collect all the store objects
            self.stores_list.append(agent)

        # Initialize all household agents and add them to the scheduler and the Geospace
        for house in self.households:
            agent = Household(
                self, 
                house[0], #id
                house[1], #polygon
                house[2], #income
                house[3], #household_size
                house[4], #vehicles
                house[5], #number of workers
                house[6], #walking_time
                house[7], #biking_time
                house[8], #transit_time
                house[9], #driving_time
                SEARCHRADIUS,
                CRS)
            self.schedule.add(agent)
            self.space.add_agents(agent)
        


        self.datacollector = DataCollector(
            #model_reporters={"Average mfai": "avg_mfai"}#,
            agent_reporters={
                "Geometry": "geometry",
                "Income": "income", 
                "Household Size": "household_size", 
                "Vehicles":  "vehicles", 
                "Number of Workers":  "number_of_workers",
                "Stores within 1 Mile" :  "num_store_within_mile", 
                "Closest Store (Miles)" :  "distance_to_closest_store", 
                "Rating for Distance to Closest Store" :   "rating_distance_to_closest_store", 
                "Rating for Number of Stores within 1.0 Miles" :  "rating_num_store_within_mile", 
                "Ratings Based on Num of Vehicle" :  "rating_based_on_num_vehicles",
                "Transit time":  "transit_time",
                "Walking time":  "walking_time",
                "Biking time":  "biking_time",
                "Driving time":  "driving_time",
                "Food Access Score" :  "mfai",
                "Color": "color"}
        )
        self.datacollector.collect(self)
    def get_stores(self):
        return self.stores
    def get_households(self):
        print(len(self.datacollector.get_agent_vars_dataframe().tail(len(self.households))))
        return self.datacollector.get_agent_vars_dataframe().tail(len(self.households))
    def reset_stores(self):
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        cursor = connection.cursor()

        # Execute the SQL query
        cursor.execute("SELECT * FROM food_stores;")

        # Fetch all rows from the executed query
        self.stores = cursor.fetchall()

        cursor.close()
        connection.close()

         # Initialize all store agents and add them to the GeoSpace
        index_count = 0
        self.stores_list.clear()
        for store in self.stores:
            agent = Store(
                self, 
                index_count + len(self.households), 
                store[2], #name
                store[0], #shop
                store[1] #geo
                )
            index_count+=1
            self.space.add_agents(agent) 
            # Initializing empty list to collect all the store objects
            self.stores_list.append(agent)

        return None
    
    def step(self) -> None:

        """
        Step function. Runs one step of the GeoModel.
        """
        self.schedule.step() 
        self.datacollector.collect(self)
        