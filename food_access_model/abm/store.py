from mesa_geo import GeoAgent
from shapely.geometry import Point
from pyproj import Transformer

_TO_3857 = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

class Store(GeoAgent):
    """
    Represents a Store. Extends the mesa_geo GeoAgent class.
    """
    def __init__(self, model,
                 id: int,
                 name: str = None,
                 type: str = None,
                 longitude: float = None,
                 latitude: float = None) -> None:
        """
        Initialize the Store Agent.

        Args:
            model (GeoModel): model from mesa that places stores on a GeoSpace
            id (int): store's unique id
            name (String): Name of grocery store
            type (String): can be one of [convenience, supermarket, butcher, wholesale,
                                          farm, greengrocer, health_food, grocery]
            longitude (float): EPSG:4326 longitude coordinate
            latitude (float): EPSG:4326 latitude coordinate
        """
        # Convert 4326 to 3857 for in-memory spatial math
        x_3857, y_3857 = _TO_3857.transform(longitude, latitude) 
        point = Point(x_3857, y_3857)
        super().__init__(id, model, point, "epsg:3857")
        self.type = type
        self.name = name
        self.longitude = longitude  # original 4326 for reference
        self.latitude = latitude    # original 4326 for reference
