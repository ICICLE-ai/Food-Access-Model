from dataclasses import dataclass
from typing import Tuple, Optional
from pydantic import BaseModel
import json
from decimal import Decimal

MAX_RENDERING_POINTS = 10000
MIN_ZOOM_FOR_POLYGONS = 14
MIN_ZOOM_FOR_SUPERCLUSTERS = 13
MAX_SUPERCLUSTER_POINTS = 30

@dataclass
class Bound:
    lat: float
    lng: float


@dataclass
class RectangleBound:
    north_east: Bound
    south_west: Bound


@dataclass
class GetDrawableAgentsRequest:
    step_number: int
    max_elements: int
    center: Tuple[float, float]
    zoom: float
    bounds: RectangleBound


class DrawableAgent(BaseModel):
    id: str = None
    agent_type: str = None
    geometry: Optional[str] = None
    centroid_xy: Optional[Tuple[float, float]] = None
    properties: Optional[dict] = None

class DrawableAgentEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DrawableAgent):
            return {
                "id": obj.id,
                "agent_type": obj.agent_type,
                "geometry": obj.geometry,
                "centroid_xy": [obj.centroid_xy[0], obj.centroid_xy[1]],
                "properties": obj.properties,
            }
        return super().default(obj)


def parse_drawable_agents_request(
    step_number: int,
    lat: float,
    lng: float,
    north_east_lat: float,
    north_east_lng: float,
    south_west_lat: float,
    south_west_lng: float,
    zoom: float,
    max_elements: int = MAX_RENDERING_POINTS,
) -> GetDrawableAgentsRequest:
    return GetDrawableAgentsRequest(
        step_number=step_number,
        max_elements=max_elements,
        center=(lng, lat),
        zoom=zoom,
        bounds=RectangleBound(
            north_east=Bound(lat=north_east_lat, lng=north_east_lng),
            south_west=Bound(lat=south_west_lat, lng=south_west_lng),
        ),
    )
