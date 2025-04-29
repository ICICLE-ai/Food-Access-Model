from mesa_geo import GeoAgent
import shapely
import random


class Household(GeoAgent):

    """
    Represents one Household. Extends the mesa_geo GeoAgent class. The step function
    defines the behavior of a single household on each step through the model.
    """

    def __init__(self, model, id: int, polygon, income, household_size,vehicles,number_of_workers,walking_time,biking_time,transit_time,driving_time,search_radius: int, crs: int):
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
        self.walking_time = walking_time
        self.biking_time = biking_time
        self.transit_time = transit_time
        self.driving_time = driving_time
        
        f,f,self.distance_to_closest_store,f = self.closest_cspm_and_spm()
        self.rating_num_store_within_mile = "A"
        self.rating_distance_to_closest_store = "A"
        self.rating_based_on_num_vehicles = "A"
        self.num_store_within_mile = self.stores_with_1_miles() 
        self.mfai = self.get_mfai()
        self.color = self.get_color()


    def get_color(self):
        #change to chosen variable
        value = self.mfai
        """
        helper function for agent_portrayal. Assigns a name to a value on a red-yellow-green scale.

        Args:
            - value: the value that is to be parsed into hex color.
        """
        #used to change how dark the color is
        top_range = 255

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
        self.rating_evaluation(total)
        return total 
    
    def closest_cspm_and_spm(self):
        cspm = None
        cspm_distance = 10000000
        spm = None
        spm_distance = 10000000
        for store in self.model.stores_list: 
            distance = self.model.space.distance(self,store)
            distance = round(distance/1609.34,2)
            if store.type == "supermarket":
                if distance <= spm_distance:
                    spm = store
                    spm_distance = distance
            else:
                if distance <= cspm_distance:
                    cspm = store
                    cspm_distance = distance
        return cspm, spm, spm_distance, cspm_distance
    
    def get_mfai(self):
        #calculate mfai
        cspm, spm,f,f = self.closest_cspm_and_spm()
        food_avail = list()
        for i in range(7):
            chance_of_choosing_spm = int(((self.vehicles*10)+(self.income/200000)*80))
            store = random.choices([cspm,spm], [(chance_of_choosing_spm-100)*-1,chance_of_choosing_spm], k=1)[0]
            fsa = 0
            if store is not None and store.type == "supermarket":
                fsa = 100
            else:
                fsa = 55
            if self.vehicles == 0:
                fsa = fsa*0.8
            fsa = fsa*0.85+fsa*0.25*abs(1-self.distance_to_closest_store)
            food_avail.append(fsa)

        return int(sum(food_avail)/700*100)

    def step(self) -> None:
        self.mfai = self.get_mfai()
        self.color = self.get_color()
        f,spm,self.distance_to_closest_store,f = self.closest_cspm_and_spm()
        self.num_store_within_mile = self.stores_with_1_miles()
        return None
