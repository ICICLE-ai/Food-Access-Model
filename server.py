from geo_model import GeoModel
from custom_map_visualization import MapModule
from mesa.visualization import ModularServer, Slider, ChartModule
from agent_visualization import agent_portrayal
import pandas as pd


model_params = {}

#Create Map visualization of Stores and households
map_vis = MapModule(agent_portrayal)

"""
Create chart to track mfai score
chart = ChartModule(
    [{"Label": "Average mfai", "Color": "Black"}],
    data_collector_name='datacollector'
)
"""

#Start server on port 8080
server = ModularServer(
    GeoModel,
    [map_vis],
    "Food Access Strategy Simulation",
    model_params,
)
print("running")
server.launch(8080)