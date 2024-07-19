from geo_model import GeoModel
from custom_map_visualization import MapModule
from mesa.visualization import ModularServer, Slider, ChartModule
from agent_visualization import agent_portrayal
import pandas as pd

stores = pd.read_csv("data/stores.csv")
households = pd.read_csv("data/households.csv")

# Create a dictionary to hold model parameters
model_params = {
    "stores": stores,  # Pass the stores DataFrame to the model
    "households": households,  # Pass the households DataFrame to the model
}

# Creates an instance of MapModule using agent_portrayal.
map_vis = MapModule(agent_portrayal)


"""
Create chart to track mfai score
chart = ChartModule(
    [{"Label": "Average mfai", "Color": "Black"}],
    data_collector_name='datacollector'
)
"""

# Set up and start the Mesa server for the simulation if the user wants to run the simulation
server = ModularServer(
    GeoModel, # The model class to run
    [map_vis], # List of visualization modules to use
    "Food Access Strategy Simulation",  # Title of the simulation
    model_params, # Dictionary of parameters to pass to the model
)

print("running") 
server.launch(8080) # Launch the server on port 8080