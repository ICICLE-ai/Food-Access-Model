#helper functions for api_server.py

from pydantic import BaseModel
from shapely import Polygon
from pyproj import Transformer
import math


class StoreInput(BaseModel):
    """
    Data model for adding a store.

    Attributes:
        name (str): The name of the store.
        category (str): The category/type of the store (e.g., supermarket, grocery).
        longitude (str): The longitude coordinate of the store location.
        latitude (str): The latitude coordinate of the store location.
    """   
    name: str
    category: str
    longitude: str 
    latitude: str

def convert_centroid_to_polygon(latitude, longitude, type):
    """
    Converts latitude and longitude into a polygon to display a store's area.
    
    Transforms coordinates from EPSG:4326 to EPSG:3857 and returns a polygon
    shape according to the store type. 
    
    Args:
        latitude (str): Store's latitude.
        longitude (str): Store's longitude.
        type (str): Store type (e.g., "supermarket", "grocery").

    Returns:
        str: Polygon in Well-Known Text format.
    """
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
