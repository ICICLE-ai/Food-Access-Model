from mesa_geo import GeoAgent
from shapely.geometry import Polygon,Point
from shapely.ops import transform
from shapely.wkt import loads
import pyproj
from store import Store
import random
random.seed(1)


class Household(GeoAgent):

    """
    Represents one Household. Extends the mesa_geo GeoAgent class. The step function
    defines the behavior of a single household on each step through the model.
    """

    def __init__(self, model, id: int, polygon, income, household_size,vehicles,number_of_workers, stores_list,search_radius: int, crs: int):
        """
        Initialize the Household Agent.

        Args:
            - model (GeoModel): model from mesa that places Households on a GeoSpace
            - id: id number of agent
            - polygon (Polygon): a shapely polygon that represents a houshold on the map
            - income (int): total income of the household
            - household_size (int): total members in the household
            - vehicles (int): total vechiles in the household
            - number_of_workers (int): total working members (having job) in the household
            - stores_list : List containing all the stores with their attributes
            - search_radius (int): how far to search for stores
            - crs (string): constant value (i.e.3857),used to map households on a flat earth display
        """

        #Transform shapely coordinates to mercator projection coords
        polygon = loads(polygon)
        
        # Setting argument values to the passed parameteric values.
        super().__init__(id,model,polygon,crs)
        self.income = income
        self.search_radius = search_radius
        self.household_size = household_size
        self.vehicles = vehicles
        self.number_of_workers = number_of_workers
        self.stores_list = stores_list

        # Variable to store number of stores within 1 mile of a Household
        self.num_store_within_mile = self.stores_with_1_miles() 
        
        
    def stores_with_1_miles (self):
        total = 0 
        for store in self.stores_list: 
         distance = self.model.space.distance(self,store)
         if distance <= 1609.34:
          total += 1 
        return total 

    def step(self) -> None:
       return None