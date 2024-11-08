#helper functions for api_server.py

from pydantic import BaseModel
from shapely import Polygon


class StoreInput(BaseModel): #used for adding a store
    name: str
    category: str
    longitude: str 
    latitude: str

def convert_centroid_to_polygon(latitude, longitude):
    offset = 0.00002
    # Define the four corners of the bounding box
    latitude = float(latitude)
    longitude = float(longitude)
    top_left = (longitude - offset, latitude + offset)
    top_right = (longitude + offset, latitude + offset)
    bottom_right = (longitude + offset, latitude - offset)
    bottom_left = (longitude - offset, latitude - offset)
    
    # Create a polygon from these points
    bounding_box = Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
    
    return bounding_box.wkt
