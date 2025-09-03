import os

from mesa import Model, DataCollector  # Base class for GeoModel
from mesa.time import RandomActivation  # Used to specify that agents are run randomly within each step
from mesa_geo import GeoSpace  # GeoSpace that houses agents
import psycopg2
from typing import List, Any

from food_access_model.abm.store import Store  # Store agent class
from food_access_model.abm.household import Household  # Household agent class


PASS = os.getenv("PASS")
USER = os.getenv("USER")
NAME = os.getenv("NAME")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

SEARCHRADIUS = 500
CRS = "3857"  # constant value (i.e.3857),used to map households on a flat earth display

class GeoModel(Model):
    """
    Geographical model that extends the mesa Model base class. This class initializes the store and household agents
    and then places the agents in the mesa_geo GeoSpace, which allows the Household agents to calculate distances between
    themselves and Store Agents.
    """

    def __init__(self, households: List[Any], stores: List[Any]) -> None:
        """
        Initialize the Model, initialize all agents and, add all agents to GeoSpace and Model.

        Parameters:
            stores: dataframe containing data for store agents
            households: dataframe containing data for household agents
        """
        super().__init__() 

        self.raw_step_number = 0

        # print("Initializing GeoModel...\n", flush=True)
        # print("# Of households: ", len(households), flush=True)
        # print("# Of stores: ", len(stores), flush=True)

        # Instead of RandomActivation or BaseScheduler, use ParallelScheduler
        # self.schedule = ParallelScheduler(self)
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
                store['store_id'],
                store['name'],
                store['shop'],
                store['geometry']
            )
            index_count += 1
            self.space.add_agents(agent)
            # Initializing empty list to collect all the store objects
            self.stores_list.append(agent)

        # Initialize all household agents and add them to the scheduler and the Geospace
        for house in self.households:
            agent = Household(
                self,
                house['id'],  # id
                house['Geometry'],  # polygon
                house['Income'],  # income
                house['Household Size'],  # household_size
                house['Vehicles'],  # vehicles
                house['Number of Workers'],  # number of workers
                house['Walking time'],  # walking_time
                house['Biking time'],  # biking_time
                house['Transit time'],  # transit_time
                house['Driving time'],  # driving_time
                SEARCHRADIUS,
                CRS,
                house['Closest Store (Miles)'] if 'Closest Store (Miles)' in house else 0,  # distance_to_closest_store
                house['Stores within 1 Mile'] if 'Stores within 1 Mile' in house else 0,  # num_store_within_mile
                house['Food Access Score'] if 'Food Access Score' in house else 0,  # mfai
                house['Color'] if 'Color' in house else None  # color (Coming from a previous model's step execution)
            )
            self.schedule.add(agent)
            self.space.add_agents(agent)

        self.datacollector = DataCollector(
            # model_reporters={"Average mfai": "avg_mfai"},
            agent_reporters={
                "Type": "type",
                "Geometry": "raw_geometry",
                "Income": "income",
                "Household Size": "household_size",
                "Vehicles": "vehicles",
                "Number of Workers": "number_of_workers",
                "Stores within 1 Mile": "num_store_within_mile",
                "Closest Store (Miles)": "distance_to_closest_store",
                "Rating for Distance to Closest Store": "rating_distance_to_closest_store",
                "Rating for Number of Stores within 1.0 Miles": "rating_num_store_within_mile",
                "Ratings Based on Num of Vehicle": "rating_based_on_num_vehicles",
                "Transit time": "transit_time",
                "Walking time": "walking_time",
                "Biking time": "biking_time",
                "Driving time": "driving_time",
                "Food Access Score": "mfai",
                "Color": "color",
            }
        )
        self.datacollector.collect(self)

    def set_step_number(self, step_number: int) -> None:
        """
        Sets the number of steps the model has taken

        Parameters:
            step_number (int): A numerical value representing the step number
        """
        self.raw_step_number = step_number

    def get_stores(self) -> list:
        """
        Gets all stores in the model

        Returns:
            list: A list of store objects
        """
        return self.stores

    def get_households(self):
        """
        Gets data about the most recent household agents and their variables

        Returns:
            pandas.Dataframe: A dataframe containing agents and their parameters
        """
        print(len(self.datacollector.get_agent_vars_dataframe().tail(len(self.households))))
        return self.datacollector.get_agent_vars_dataframe().tail(len(self.households))

    def step(self) -> None:
        """Runs one step of the GeoModel and collects data after each step"""
        self.schedule.step()
        self.datacollector.collect(self)
