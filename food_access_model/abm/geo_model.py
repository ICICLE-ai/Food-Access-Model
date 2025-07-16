from mesa import Model, DataCollector #Base class for GeoModel
from mesa.time import RandomActivation #Used to specify that agents are run randomly within each step
from mesa_geo import GeoSpace #GeoSpace that houses agents
from food_access_model.abm.store import Store # Store agent class
from food_access_model.abm.household import Household # Household agent class
from typing import List, Any
import psycopg2
import os

PASS = os.getenv("PASS")
USER = os.getenv("USER")
NAME = os.getenv("NAME")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

SEARCHRADIUS = 500
CRS = "3857"

class GeoModel(Model):
    """
    Geographical model that extends the mesa Model base class. This class initializes the store and household agents
    and then places the agents in the mesa_geo GeoSpace, which allows the Household agents to calculate distances between
    between themselves and Store Agents.
    """

    def __init__(self, households: List[Any], stores: List[Any]):
        """
        Initialize the Model, intialize all agents and, add all agents to GeoSpace and Model.

        Args:
            - stores: dataframe containing data for store agents
            - households: dataframe containing data for household agents
        """
        super().__init__() 

        self.raw_step_number = 0
        
        # print("Initializing GeoModel...\n", flush=True)
        # print("# Of households: ", len(households), flush=True)
        # print("# Of stores: ", len(stores), flush=True)
        
        # Instead of RandomActivation or BaseScheduler, use ParallelScheduler
        #self.schedule = ParallelScheduler(self)
        # Create new GeoSpace to contain agents
        self.space = GeoSpace(warn_crs_conversion=False) 
        # Specify that agents should be activated randomly during each step
        self.schedule = RandomActivation(self) 
        # Initializing empty list to collect all the store objects
        self.stores_list = []
        
        # start_time = time.time()
        
        self.stores = stores
        self.households = households
        
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
                CRS,
                house[10] if len(house) > 10 else 0, #distance_to_closest_store (Coming from a previous model's step execution)
                house[11] if len(house) > 11 else 0, # num_store_within_mile (Coming from a previous model's step execution)
                house[12] if len(house) > 12 else 0, # mfai (Coming from a previous model's step execution)
                house[13] if len(house) > 13 else None # color (Coming from a previous model's step execution)
            )
            self.schedule.add(agent)
            self.space.add_agents(agent)
        


        self.datacollector = DataCollector(
            #model_reporters={"Average mfai": "avg_mfai"},
            agent_reporters={
                "Type": "type",
                "Geometry": "raw_geometry",
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
                "Color": "color",
                "Centroid": "centroid"
            }
       )
        self.datacollector.collect(self)
        
    def set_step_number(self, step_number):
        self.raw_step_number = step_number
            
    def get_stores(self):
        return self.stores
    def get_households(self):
        print(len(self.datacollector.get_agent_vars_dataframe().tail(len(self.households))))
        return self.datacollector.get_agent_vars_dataframe().tail(len(self.households))
    def reset_stores(self):
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            host=HOST,
            database=NAME,
            user=USER,
            password=PASS,
            port=PORT
        )
        cursor = connection.cursor()

        # Execute the SQL query
        cursor.execute("SELECT * FROM food_stores;")

        # Fetch all rows from the executed query
        self.stores = cursor.fetchall()

        #cursor.close()
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
        
