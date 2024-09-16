from mesa_geo import GeoAgent
import shapely


class Household(GeoAgent):

    """
    Represents one Household. Extends the mesa_geo GeoAgent class. The step function
    defines the behavior of a single household on each step through the model.
    """

    def __init__(self, model, id: int, polygon, income, household_size,vehicles,number_of_workers, search_radius: int, crs: int):
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
        polygon = shapely.wkt.loads(polygon)
        # Setting argument values to the passed parameteric values.
        super().__init__(id,model,polygon,crs)
        self.income = income
        self.search_radius = search_radius
        self.household_size = household_size
        self.vehicles = vehicles
        self.number_of_workers = number_of_workers

        self.distance_to_closest_store = 100000
        self.rating_num_store_within_mile = "A"
        self.rating_distance_to_closest_store = "A"
        self.rating_based_on_num_vehicles = "A"
        # Variable to store number of stores within 1 mile of a Household
        self.num_store_within_mile = self.stores_with_1_miles() 
        # print(self.stores_with_1_miles())


    def rating_evaluation(self,total):
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
        
    def stores_with_1_miles (self):
        total = 0 
        for store in self.model.stores_list: 
         distance = self.model.space.distance(self,store)
         if distance <= 1609.34:
          total += 1 
          if self.distance_to_closest_store > distance: 
             self.distance_to_closest_store = round((distance)/1609.34,2)
        self.rating_evaluation(total)
        return total 

    def step(self) -> None:
       return None