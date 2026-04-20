from mesa_geo import GeoAgent
from pyproj import Transformer
from shapely.geometry import Point
import shapely
import random

_TO_3857 = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)     

class Household(GeoAgent):
    """
    Represents one Household. Extends the mesa_geo GeoAgent class. The step function
    defines the behavior of a single household on each step through the model.
    """
    def __init__(self, model, id: int, income: int, household_size: int, vehicles: int, number_of_workers: int, walking_time: int, biking_time: int, transit_time: int, driving_time: int, search_radius: int, crs: str, distance_to_closest_store: float = None, num_store_within_mile: int = None, mfai: int = None, color: str= None) -> None:
        """
        Initialize the Household Agent.

        Args:
            - model (GeoModel): model from mesa that places Households on a GeoSpace
            - id: id number of agent
            - income (int): total income of the household
            - household_size (int): total members in the household
            - vehicles (int): total vechiles in the household
            - number_of_workers (int): total working members (having job) in the household
            - stores_list : List containing all the stores with their attributes
            - search_radius (int): how far to search for stores (default 500)
        """
        # Keep original 4326 WKT for DB writes
        self.raw_geometry = polygon 

        # Reproject from 4326 to 3857 for in-memory spatial math, but the original 4326 geometry is kept in self.raw_geometry was saved for database writes
        point_4326 = shapely.wkt.loads(polygon)
        x_3857, y_3857 = _TO_3857.transform(point_4326.x, point_4326.y)
        point_3857 = Point(x_3857, y_3857)
        
        # Setting argument values to the passed parameteric values.
        super().__init__(id, model, point_3857, "epsg:3857")
        self.income = income
        self.search_radius = search_radius
        self.household_size = household_size
        self.vehicles = vehicles
        self.number_of_workers = number_of_workers
        self.walking_time = walking_time
        self.biking_time = biking_time
        self.transit_time = transit_time
        self.driving_time = driving_time
        self.type="household"

        #f,f,self.distance_to_closest_store,f = self.closest_cspm_and_spm()
        self.rating_num_store_within_mile = "A"
        self.rating_distance_to_closest_store = "A"
        self.rating_based_on_num_vehicles = "A"

        self.distances_map =None
        self.distance_to_closest_store = distance_to_closest_store
        self.num_store_within_mile = num_store_within_mile
        self.mfai = mfai #MFAI (monthly food access index)
        self.color = color
        self.has_vehicles = self.vehicles > 0
        self.resources = self.has_resources()
        self.monthly_trips = self.get_monthly_trip_count()

    def get_color(self) -> str:
        """
        Helper function for agent_portrayal. Use store's MFAI to assign a color on the red-yellow-green scale.

        Returns:
            str: hex value correlating to a color
        """
        # constants
        MAX_RGB = 255

        # change to chosen variable
        value = self.mfai #the value that is to be parsed into hex color.

        # used to change how dark the color is
        top_range = MAX_RGB

        # Normalize value to a range of 0 to 1
        normalized = abs(((value)-40)/60) #this is hardcoded

        # If value is too low just return red
        if normalized < 0:
            red = top_range
            green = 0
            blue = 0
        # Calculate the red, green, and blue components
        elif normalized < 0.5:
            # Interpolate between red (255, 0, 0) and yellow (255, 255, 0)
            red = top_range
            green = int(top_range * (normalized * 2))
            blue = 0
        else:
            # Interpolate between yellow (255, 255, 0) and green (0, 255, 0)
            red = int(top_range * (2 - 2 * normalized))
            green = top_range
            blue = 0

        gray = 128
        desaturation_factor = .25

        # Desaturating respective colors (RED,GREEN,BLUE)
        red = int(red * (1 - desaturation_factor) + gray * desaturation_factor)
        green = int(green * (1 - desaturation_factor) + gray * desaturation_factor)
        blue = int(blue * (1 - desaturation_factor) + gray * desaturation_factor)

        # Convert RGB to hexadecimal
        hex_color = f"#{red:02x}{green:02x}{blue:02x}"

        return hex_color

    def rating_evaluation(self, total: int) -> None:
        """
        Assigns a rating of A,B,C,D to the number of stores within a mile, distance to the closest store,
        and the number of vehicles and workers

        Parameters:
            total (int): number of stores within a mile of the household
        """
        if total < 2:
            self.rating_num_store_within_mile = "D"
        if total < 5 and total >= 2:
            self.rating_num_store_within_mile = "C"    
        if total < 10 and total >= 5:
            self.rating_num_store_within_mile = "B"  
        if self.distance_to_closest_store > 2.00: 
            self.rating_distance_to_closest_store  = "D"  
        if self.distance_to_closest_store > 1.00 and self.distance_to_closest_store <= 2.00: 
            self.rating_distance_to_closest_store  = "C"  
        if self.distance_to_closest_store > 0.50 and self.distance_to_closest_store <= 1.00: 
            self.rating_distance_to_closest_store  = "B"   
        if self.vehicles == 0:  
            self.rating_based_on_num_vehicles = "C"   
        if self.vehicles < self.number_of_workers and self.vehicles > 0: 
            self.rating_based_on_num_vehicles = "B"     

    def stores_with_1_miles (self) -> int:
        """
        Calculates the number of stores within a mile of the household

        Returns:
            int: total number of stores within a mile
        """
        total = 0 
        for store in self.model.stores_list: 
         # distance is already in miles (converted in calculate_distances)
         distance = self.distances_map[store.unique_id]
         if distance <= 1.0:
          total += 1 
        self.rating_evaluation(total)
        return total
    
    def get_closest_cspm(self) -> tuple:
        cspm = None
        cspm_distance = 10000000
        for store in self.model.stores_list:
            if store.type != "supermarket":
                distance = self.get_store_dist(store)
                if distance <= cspm_distance:
                    cspm = store
                    cspm_distance = distance
        return (cspm, cspm_distance)
    
    def get_closest_spm(self) -> tuple:
        spm = None
        spm_distance = 10000000
        for store in self.model.stores_list:
            if store.type == "supermarket":
                distance = self.get_store_dist(store)
                if distance <= spm_distance:
                    spm = store
                    spm_distance = distance
        return (spm, spm_distance)

    def has_resources(self) -> bool:
        if self.income < 10000:
            return False
        if self.household_size >= 2 and self.income < 15000:
            return False
        if self.household_size >= 3 and self.income < 25000:
            return False
        return True
    
    def get_monthly_trip_count(self) -> int:
        if self.resources:
            if self.has_vehicles:
                return 7
            else:
                return 8
        else:
            return 6

    # chance of choosing a close spm is just hard code val 0.8
    def chance_of_choosing_spm(self, spm_dist, cspm_dist) -> float:
        if spm_dist < cspm_dist:
            return 0.8
        
        if self.resources:
            if self.has_vehicles:
                return 0.76
            else:
                return 0.72
        else:
            if self.has_vehicles:
                return 0.64
            else:
                return 0.6
            
    def get_store_dist(self, store) -> float:
        return self.distances_map[store.unique_id]
    
    # returns store object
    def choose_store(self, spm, cspm, spm_dist, cspm_dist) -> object:
        if spm is None:
            return cspm
        if cspm is None:
            return spm

        spm_chance = self.chance_of_choosing_distant_spm(spm_dist, cspm_dist)

        #randomly choose based off chances
        return random.choices([cspm, spm], [(1 - spm_chance), spm_chance], k = 1)[0]

    def get_mfai(self) -> int:
        """
        Calculates the MFAI (monthly food access index)

        Parameters:
            cspm (object): the closest market to the household that's not a supermarket
            spm (object): the closest supermarket to the household

        Returns:
            int: the mfai value
        """
        # closest cspm/spm
        closest_cspm, cspm_dist = self.get_closest_cspm()
        closest_spm, spm_dist = self.get_closest_spm()

        food_avail = list()
        for i in range(self.monthly_trips):
            # randomly select the closest spm/cspm
            store = self.choose_store(closest_spm, closest_cspm, spm_dist, cspm_dist)

            if store is not None and store.type == "supermarket":
                fsa = 95
            else:
                fsa = 55

            food_avail.append(fsa)
        return sum(food_avail) / len(food_avail)

    def calculate_distances(self)-> None:
        """
        Creates dictionary with key (indicating the store) and value (indicating the distance from the household to
        that store)
        """
        METERS_IN_MILE = 1609.34
        self.distances_map = dict()
        for store in self.model.stores_list: 
            agent_unique_id  = store.unique_id
            distance = self.model.space.distance(self,store)
            distance = round(distance/METERS_IN_MILE,2)
            self.distances_map[agent_unique_id] = distance 

    def step(self) -> None:
        """
        Recalculates the households values after a step in the simulation
        """
        if self.distances_map is None:
            self.calculate_distances()
        # find spm for get_color and rating_evaluation methods (cspm and spm not needed for mfai method anymore)
        spm, spm_dist = self.get_closest_spm()
        if spm is not None:
            self.distance_to_closest_store = spm_dist

        self.num_store_within_mile = self.stores_with_1_miles()
        self.mfai = self.get_mfai()
        self.color = self.get_color()

        return None
