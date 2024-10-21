from mesa import Model, DataCollector #Base class for GeoModel
from mesa.time import RandomActivation #Used to specify that agents are run randomly within each step
from mesa_geo import GeoSpace #GeoSpace that houses agents
import pandas as pd
from store import Store # Store agent class
from household import Household # Household agent class
import psycopg2
from config import USER, PASS, HOST, NAME, PORT
import logging
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        stores = cursor.fetchall()
        logging.info(f"csv households: {stores}")
        # Execute the SQL query
        cursor.execute("SELECT * FROM households;")

        # Fetch all rows from the executed query
        households = cursor.fetchall() 
        #logging.info(f"Fetched households: {households}")
        #For Testing
        households = pd.read_csv("./old_benchmarking/testing_data_columbus_cols.csv").values.tolist()
        logging.info(f"csv households: {str(households)}")
        #stores = pd.read_csv("./old_benchmarking/stores_columbus.csv", header=None)
        logging.info(f"stores households: {stores}")
        #FOR BENCHMARKING
        #households = pd.read_csv("testdata_home.csv")
        #stores = pd.read_csv("testdata_store.csv")

        # Initialize all store agents and add them to the GeoSpace
        index_count = 0
        for store in stores:
            agent = Store(
                self, 
                index_count + len(households), 
                store[2], #name
                store[0], #shop
                store[1] #geo
                )
            index_count+=1
            self.space.add_agents(agent) 
            # Initializing empty list to collect all the store objects
            self.stores_list.append(agent)

        # Initialize all household agents and add them to the scheduler and the Geospace
        for house in households:
            logging.info(f"the subscriptable {households}")
            agent = Household(
                self, 
                house[0],
                house[1], 
                house[2],
                house[3],
                house[4],
                house[5],
                SEARCHRADIUS,
                CRS)
            self.schedule.add(agent)
            self.space.add_agents(agent)

        #self.datacollector = DataCollector(
        #    model_reporters={"Average mfai": "avg_mfai"}#,
        #    #agent_reporters={"Mfai": "mfai"}
        #)
        #self.datacollector.collect(self)

    def step(self) -> None:

        """
        Step function. Runs one step of the GeoModel.
        """
        self.schedule.step() 
        #self.datacollector.collect(self)
        