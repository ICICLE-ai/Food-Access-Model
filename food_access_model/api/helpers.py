#helper functions for api_server.py

from pydantic import BaseModel
from shapely import Polygon
from pyproj import Transformer
import math


class StoreInput(BaseModel): #used for adding a store
    name: str
    category: str
    longitude: str 
    latitude: str

def convert_centroid_to_polygon(latitude, longitude, type):
    # Define the transformation from EPSG:4326 to EPSG:3857
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

    latitude = float(latitude)
    longitude = float(longitude)

    longitude,latitude = transformer.transform(longitude,latitude)

    print(latitude,longitude)

    offset = 40
    # Define the four corners of the bounding box
    if (type == "supermarket") or (type=="grocery") or (type=="greengrocer"):
        polygon = Polygon([(longitude + 50 * math.cos(math.radians(angle)), latitude + 50 * math.sin(math.radians(angle))) for angle in range(0, 360, 60)])
    else:
        polygon = Polygon([
            (longitude, latitude + 20),           # Top vertex
            (longitude + 25, latitude - 30),      # Bottom right vertex
            (longitude - 25, latitude - 30)       # Bottom left vertex
        ])

    return polygon.wkt
