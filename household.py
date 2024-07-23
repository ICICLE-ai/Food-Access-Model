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

    def __init__(self, model, id: int, latitude, longitude, polygon, income, household_size,vehicles,number_of_workers, search_radius: int, crs: str):
        """
        Initialize the Household Agent.

        Args:
            - model (GeoModel): The model from mesa that places Households on a GeoSpace.
            - id (int): Unique identifier for the agent.
            - latitude (float): Latitude coordinate of the household.
            - longitude (float): Longitude coordinate of the household.
            - polygon (str): Well-known text (WKT) representation of the household's polygon.
            - income (float): Income level of the household.
            - household_size (int): Number of people in the household.
            - vehicles (int): Number of vehicles owned by the household.
            - number_of_workers (int): Number of workers in the household.
            - search_radius (int): Initial radius to search for stores.
            - crs (str): Coordinate reference system for the geometry.
        """

        #Transform shapely coordinates to Mercator projection coords
        polygon = loads(polygon)
        

        super().__init__(id,model,polygon,crs)
        self.income = income
        self.search_radius = search_radius
        self.latitude = latitude
        self.longitude = longitude
        self.household_size = household_size
        self.vehicles = vehicles
        self.number_of_workers = number_of_workers

    def choose_store(self, search_radius):
        """
        Helper method for step function. This method optimizes time complexity of the step function
        by recursively increasing search radius until it finds a spm and a cspm. Ideally this function will not
        recurse except in edge cases. Ultimately, this method finds and chooses store to shop at.

        TODO: Statistical analysis should be used to find the most optimal search radius and increase to search radius
        for each step. This function is the main bottleneck for speed in this model and should be optimized perfectly.

        Args:
            - search_radius: radius to search for stores.

        returns:
            - chosen_store: chosen store to shop at.
        """

        #(1) Find all agents within search radius
        closest_agents = self.model.space.get_neighbors_within_distance(self,search_radius)

        # Initialize variables to find the closest CSPM and SPM
        closest_cspm = None
        cspm_distance = search_radius
        closest_spm = None
        spm_distance = search_radius

        #Get closest cspm and closest spm
        for agent in closest_agents:
            if (isinstance(agent,Store)):
                if (agent.category == "SPM"):
                    distance = self.model.space.distance(self,agent)
                    if distance <= spm_distance:
                        spm_distance = distance
                        closest_spm = agent
                if (agent.category == "CSPM"):
                    distance = self.model.space.distance(self,agent)
                    if distance <= cspm_distance:
                        cspm_distance = distance
                        closest_cspm = agent

        #If stores could not be found with current search radius, expand search radius
        if (closest_cspm == None) or (closest_spm == None):
            return self.choose_store(search_radius+500)

        #Randomly (with weighted by closer prob or farther prob) choose to go to cspm or spm
        chosen_store = None
        distance = self.model.space.distance(self,closest_spm)
        #print(distance)
        distance = self.model.space.distance(self,closest_cspm)
        #print(distance)
        if(spm_distance>cspm_distance):
            #if farther, choose SPM with fartherprob
            chosen_store = random.choices([closest_spm,closest_cspm], weights=[self.farther_prob,1-self.farther_prob])[0]
        else:
            #if closer, choose SPM with closerprob
            chosen_store = random.choices([closest_spm,closest_cspm], weights=[self.closer_prob,1-self.closer_prob])[0]

        return chosen_store
    
    def step(self) -> None:
        """
        Defines the behavior of a single household on each step through the model. NEED TO CREATE LIST OF STORES AND ONLY SEARCH THROUGH STORES

        (1) Finds closest SPM and Closest CSPM
        (2) Calculates mfai based on chosen store's fsa score
        
        """
        #chosen_store = self.choose_store(self.search_radius)

        # calculate mfai score:
        # code returns a moving average, however mfai per month is calculated like:
        # FOR EACH VISIT: (number of visits per month given by trips_per_month)
        # Sum of Food per month += ((Store's FSA score) * (Percent that the agent can carry))
        # AT END OF MONTH
        # mfai = ((Sum of Food per month) / (Maximum MFAI)) * 100
        #food_on_this_visit = ((chosen_store.fsa*self.max_carry_percent) / self.mfai_max) * 100
        #self.mfai = (self.mfai - self.mfai/self.trips_per_month) + food_on_this_visit