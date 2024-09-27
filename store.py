from mesa_geo import GeoAgent
from shapely.geometry import Polygon, Point
import shapely


class Store(GeoAgent):
    """
    Represents a Store. Extends the mesa_geo GeoAgent class.
    """

    def __init__(self, model, id: int, name, type, geometry) -> None:
        """
        Initialize the Household Agent.

        Args:
            - model (GeoModel): model from mesa that places stores on a GeoSpace
            - id (int): store's unique id
            - name (String): Name of grocery store
            - type (String): can be either [CurbPickup, EthnicFoods, GroceRetail,HealthFoods, ShoppingService, SpecialtyFoods, WholeSale] 
            - lat (float): latitude of agent
            - lon (float): longitude of agent
            - crs (string): constant value (i.e.3857),used to map stores on a flat earth display 
        """
        polygon = shapely.wkt.loads(geometry)
        #polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
        super().__init__(id,model,polygon,"epsg:3857") # epsg:3857 is the mercator projection
        self.type = type
        self.name = name