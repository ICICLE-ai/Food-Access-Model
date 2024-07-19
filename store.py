from mesa_geo import GeoAgent
from shapely.geometry import Polygon, Point
from shapely.ops import transform, unary_union
import pyproj


class Store(GeoAgent):
    """
    Represents a Store. Extends the mesa_geo GeoAgent class.
    """

    def __init__(self,  model, id: int, name, type, lat: float, lon: float, crs) -> None:
        """
        Initialize the Household Agent.

        Args:
            - id (int): store's unique id
            - model (GeoModel): model from mesa that places stores on a GeoSpace
            - type(String) : can be either [CurbPickup, EthnicFoods, GroceRetail,HealthFoods, ShoppingService, SpecialtyFoods, WholeSale] 
            - lat (float): latitude of agent
            - lon (float): longitude of agent
            - crs (int) : constant value (i.e.3857) used to map stores on a flat earth display 
        """

        #Transform shapely coordinates to mercator projection coords
        point = Point(lat,lon)
        project = pyproj.Transformer.from_proj(
            pyproj.Proj('epsg:4326'), # source coordinate system
            pyproj.Proj('epsg:3857')) # destination coordinate system
        point = transform(project.transform, point)  # apply projection
        polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
        super().__init__(id,model,polygon,crs) # epsg:3857 is the mercator projection
        self.type = type
        self.name = name