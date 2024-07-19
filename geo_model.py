from mesa import Model, DataCollector # imports Model, the base class for Mesa models, and DataCollector, used for collecting data during model runs.
from mesa.time import RandomActivation #Used to specify that agents are run randomly within each step
from mesa_geo import GeoSpace #GeoSpace that houses agents
import pandas as pd
from store import Store # Store agent class
from household import Household # Household agent class

from constants import(
    SEARCHRADIUS,
    CRS
)

class GeoModel(Model):

    """
    Geographical model that extends the Mesa Model base class.
    This class initializes the store and household agents and then
    places the agents in the Mesa-Geo GeoSpace.
    """

    def __init__(self, stores: pd.DataFrame, households: pd.DataFrame):
        """
        Initialize the model, including agents and GeoSpace, and add all agents to GeoSpace and Model.

        Args:
            stores: dataframe containing data for store agents
            households: dataframe containing data for household agents
        """
        super().__init__()
        self.space = GeoSpace(warn_crs_conversion=False)  # Creates a new GeoSpace object for managing agents
        self.schedule = RandomActivation(self)  # Initializes the scheduler to activate agents randomly during each step.

        # Initialize store agents and add them to the GeoSpace
        for index, row in stores.iterrows():
            agent = Store(
                self,
                index + len(households),
                row["name"],
                row["type"],
                row["latitude"],
                row["longitude"],
                CRS
            )
            self.space.add_agents(agent)  # Adds the store agent to the GeoSpace.

        # Initialize household agents, add them to the scheduler, and place them in the GeoSpace
        for index, row in households.iterrows():
            agent = Household(
                self,
                row["id"],
                float(row["latitude"]),
                float(row["longitude"]),
                row["polygon"],
                row["income"],
                row["household_size"],
                row["vehicles"],
                row["number_of_workers"],
                SEARCHRADIUS,
                CRS
            )
            self.schedule.add(agent)  # Adds the household agent to the scheduler.
            self.space.add_agents(agent)  # Adds the household agent to the GeoSpace.

        #self.datacollector = DataCollector(
        #    model_reporters={"Average mfai": "avg_mfai"}#,
        #    #agent_reporters={"Mfai": "mfai"}
        #)
        #self.datacollector.collect(self)
        
    
    def step(self) -> None:
        """
        Advances the model by one step. This method updates all agents and the state of the model.
        """
        self.schedule.step()  # Advances the scheduler, updating all agents.

    #self.datacollector.collect(self)
        