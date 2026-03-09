from mesa_geo import GeoAgent
import shapely
from shapely.geometry import Point

class Store(GeoAgent):
    """
    Represents a Store. Extends the mesa_geo GeoAgent class.
    """
    def __init__(self, model,
                 id: int,
                 name: str = None,
                 type: str = None,
                 x: float = None,
                 y: float = None) -> None:
        """
        Initialize the Store Agent.

        Args:
            model (GeoModel): model from mesa that places stores on a GeoSpace
            id (int): store's unique id
            name (String): Name of grocery store
            type (String): can be one of [convenience, supermarket, butcher, wholesale,
                                          farm, greengrocer, health_food, grocery]
            x (float): EPSG:3857 x coordinate
            y (float): EPSG:3857 y coordinate
        """
        point = Point(x, y)
        super().__init__(id, model, point, "epsg:3857")
        self.type = type
        self.name = name
