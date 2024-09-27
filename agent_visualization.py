from household import Household
from store import Store


def number_to_color_word(value):
    """
    helper function for agent_portrayal. Assigns a name to a value on a red-yellow-green scale.

    Args:
        - value: the value that is to be parsed into hex color.
    """
    #used to change how dark the color is
    top_range = 255

    # Normalize value to a range of 0 to 1
    normalized = ((value)-50)/50

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

def agent_portrayal(agent):
    """
    Defines attributes for agent visualization. If agent is a houshold, it is colored on a red-green color scale, if it is a store then blue.

    Args:
        - agent: household or store to be colored red-green or blue.
    """
    portrayal = dict()

    # Adding attributes to define a Household. 
    if isinstance(agent,Household):
        # Sets the house color based on its income 
        portrayal["color"] = number_to_color_word(agent.mfai)
        # Overall Description of the house
        portrayal["description"] = [
            "Household",
            "Income: " + "{:,}".format(agent.income), 
            "Household Size: " + str(agent.household_size), 
            "Vehicles: " + str(agent.vehicles), 
            "Number of Workers: " + str(agent.number_of_workers),
            "Stores within 1.0 Miles: " + str(agent.num_store_within_mile), 
            "Distance to the Closest Store: " + str(agent.distance_to_closest_store) + " miles", 
            "Rating for Distance to Closest Store: " +  str(agent.rating_distance_to_closest_store), 
            "Rating for Number of Stores within 1.0 Miles: " + str(agent.rating_num_store_within_mile), 
            "Ratings Based on Num of Vehicle: " + str(agent.rating_based_on_num_vehicles),
            "MFAI Score: " + str(agent.mfai)
            ]

    # Adding attributes to define a Store like its color and overall description. 
    if isinstance(agent,Store):
        portrayal["color"] = "Blue"   
        portrayal["description"] = [
            "Category: " + str(agent.type),
            "Name: " + str(agent.name)]

    return portrayal 