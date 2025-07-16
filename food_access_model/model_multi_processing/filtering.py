from typing import List
import shapely
import gc
import concurrent.futures
import multiprocessing as mp
import traceback
from pyproj import Transformer
import random

from food_access_model.api.models import (
    DrawableAgent,
    GetDrawableAgentsRequest,
    RectangleBound,
    MAX_RENDERING_POINTS,
    MIN_ZOOM_FOR_POLYGONS,
)
from food_access_model.api.configs import MAX_CPUS
from food_access_model.abm.household import Household
from food_access_model.abm.store import Store
from food_access_model.model_multi_processing.superclusters import get_agent_centroid

import logging

logger = logging.getLogger(__name__)


def map_household_to_drawable_agent(household: Household) -> DrawableAgent:

    drawable_agent = DrawableAgent()
    drawable_agent.id = str(household.id)
    drawable_agent.agent_type = "household"
    drawable_agent.geometry = str(household.raw_geometry)
    drawable_agent.centroid_xy = household.centroid.x, household.centroid.y
    drawable_agent.properties = {
        "ID": str(household.id),
        "Income": household.income,
        "Household Size": household.household_size,
        "Vehicles": household.vehicles,
        "Number of Workers": household.number_of_workers,
        "Stores within 1 Mile": household.num_store_within_mile,
        "Closest Store (Miles)": household.distance_to_closest_store,
        "Rating for Distance to Closest Store": household.rating_distance_to_closest_store,
        "Rating for Number of Stores within 1.0 Miles": household.rating_num_store_within_mile,
        "Ratings Based on Num of Vehicle": household.rating_based_on_num_vehicles,
        "Transit time": household.transit_time,
        "Walking time": household.walking_time,
        "Biking time": household.biking_time,
        "Driving time": household.driving_time,
        "Food Access Score": household.mfai,
        "Color": household.get_color(),
    }

    return drawable_agent


def map_store_to_drawable_agent(store: Store) -> DrawableAgent:
    drawable_agent = DrawableAgent()
    drawable_agent.id = str(store.id)
    drawable_agent.agent_type = "store"
    drawable_agent.geometry = str(store.raw_geometry)
    drawable_agent.centroid_xy = store.centroid.x, store.centroid.y
    drawable_agent.properties = {
        "ID": str(store.id),
        "Name": store.name,
        "Type": store.type,
    }

    return drawable_agent


def process_agent_for_rendering(
    agent, should_show_polygons: bool
) -> DrawableAgent:
    drawable_agent: DrawableAgent = DrawableAgent()
    if isinstance(agent, Household):
        drawable_agent = map_household_to_drawable_agent(agent)
    elif isinstance(agent, Store):
        drawable_agent = map_store_to_drawable_agent(agent)
    else:
        raise ValueError(f"Unknown agent type: {agent.__class__.__name__}")

    if not should_show_polygons:
        del drawable_agent.geometry
    else:
        del drawable_agent.centroid_xy
    
    return drawable_agent


def preprocess_agents_for_rendering(agents, should_show_polygons: bool) -> list[DrawableAgent]:

    logger.debug(f"Preprocessing agents for rendering")
    logger.debug(f"Agents list length: {len(agents)}")
    logger.debug(f"Should show polygons: {should_show_polygons}")

    list_of_drawable_agents: List[DrawableAgent] = []
    for agent in agents:

        drawable_agent = process_agent_for_rendering(agent, should_show_polygons)

        list_of_drawable_agents.append(drawable_agent)

    return list_of_drawable_agents


def quick_bbox_check(agent, bbox_bounds):
    """Quick bounding box check before expensive contains() operation."""
    if not hasattr(agent, "centroid") or agent.centroid is None:
        return False

    point = agent.centroid
    if hasattr(point, "x") and hasattr(point, "y"):
        return (
            bbox_bounds[0] <= point.x <= bbox_bounds[2]
            and bbox_bounds[1] <= point.y <= bbox_bounds[3]
        )
    return False


def _filter_agents_batch(batch_info):
    """Filter a batch of agents by bounds with error handling and quick bbox check."""
    try:
        start, sw_x, sw_y, ne_x, ne_y, batch_centroids = batch_info

        bbox = shapely.box(sw_x, sw_y, ne_x, ne_y)
        bbox_bounds = bbox.bounds

        # Two-stage filtering: quick bbox check first, then precise contains()
        filtered_batch_idxs = []
        for idx, centroid in enumerate(batch_centroids):
            # if not quick_bbox_check(centroid):
            #     continue
            point = shapely.Point(centroid[0], centroid[1])

            if not quick_bbox_check(point, bbox_bounds):
                continue

            if not bbox.contains(point):
                continue

            target_idx = idx + start
            filtered_batch_idxs.append(target_idx)

        # Force garbage collection to free memory
        gc.collect()
        return filtered_batch_idxs

    except Exception as e:
        logger.error(f"Error filtering batch: {str(e)}")
        return []


def serial_filter_by_bounds(agents, rectangle_bound: RectangleBound) -> List:
    """
    Filter agents by bounds with serial processing and optimizations.

    Args:
        agents: List of agents to filter
        rectangle_bound: Bounding rectangle to filter by
    """

    logger.info(f"Filtering {len(agents)} agents by bounds")

    # Web Mercator to EPSG:3857 with custom y_0=27445 to match frontend config
    transformer = Transformer.from_crs(
        "+proj=longlat +datum=WGS84 +no_defs",
        "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=27445 +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    )

    # Transform bbox corners from lat/lng to Web Mercator (EPSG:3857) with custom y_0
    sw_x, sw_y = transformer.transform(
        rectangle_bound.south_west.lng, rectangle_bound.south_west.lat
    )
    ne_x, ne_y = transformer.transform(
        rectangle_bound.north_east.lng, rectangle_bound.north_east.lat
    )

    logger.debug(f"SW: {sw_x}, {sw_y}")
    logger.debug(f"NE: {ne_x}, {ne_y}")

    bbox = shapely.box(sw_x, sw_y, ne_x, ne_y)
    bbox_bounds = bbox.bounds  # (minx, miny, maxx, maxy) for quick checks
    logger.debug(f"BBOX: {bbox}")
    logger.debug(f"BBOX bounds: {bbox_bounds}")

    # Two-stage filtering for serial processing too
    return [agent for agent in agents if bbox.contains(agent.centroid)]


def filter_by_bounds_parallel(
    agents, rectangle_bound: RectangleBound, max_elements: int = MAX_RENDERING_POINTS, zoom: float = 10
) -> List:
    """
    Filter agents by bounds with parallel processing and optimizations.

    Args:
        agents: List of agents to filter
        rectangle_bound: Bounding rectangle to filter by

    Returns:
        List of filtered agents
    """
    
    max_elements = min(MAX_RENDERING_POINTS, min(max_elements, len(agents)))
    # filtered_agents_ids = []
    if zoom is None:
        should_show_polygons = False
    else:
        should_show_polygons = zoom > MIN_ZOOM_FOR_POLYGONS
    
    if not agents:
        logger.info(f"Agents list is empty, returning empty list")
        return []

    if rectangle_bound is None:
        logger.info(f"Rectangle bound is None, returning all agents")
        if len(agents) < max_elements:
            return preprocess_agents_for_rendering(agents, should_show_polygons)
        else:
            # gen max_elements random numbers between 0 and len(agents)
            logger.info(f"Sampling {max_elements} random indices from {len(agents)} agents")
            random_indices = random.sample(range(len(agents)), max_elements)   
            return preprocess_agents_for_rendering([agents[i] for i in random_indices], should_show_polygons)


    if len(agents) < max_elements:
        logger.info(
            f"Using serial processing for small dataset with {len(agents)} agents"
        )
        filtered_agents = serial_filter_by_bounds(agents, rectangle_bound)
        return preprocess_agents_for_rendering(filtered_agents, should_show_polygons)

    logger.info("Using parallel processing for large dataset")

    # Create batches
    num_cores = min(mp.cpu_count(), MAX_CPUS)
    # NOTE: Since we are using a hpc cluster, sometimes we have more cpus than
    # allocated, so if we use only cpu_count we will face issues
    
    # num_cores = mp.cpu_count()- 1  # Don't use all CPUs
    batch_size = max(
        1000, len(agents) // (num_cores * 2)
    )  # Ensure reasonable batch sizes
    chunk_size = batch_size

    logger.debug(f"Using {num_cores} cores for filtering by bounds")
    logger.debug(f"Using batch size: {batch_size}")

    all_centroids = []
    for i in range(0, len(agents), chunk_size):
        chunk = agents[i : i + chunk_size]
        chunk_centroids = [get_agent_centroid(agent) for agent in chunk]
        all_centroids.extend(chunk_centroids)

        # TODO: Force garbage collection after each chunk. CHeck safeness
        del chunk
        gc.collect()

    logger.info(f"Centroids[0]: {all_centroids[0]}")

    batches = [
        (i, min(i + batch_size, len(all_centroids)))
        for i in range(0, len(all_centroids), batch_size)
    ]

    logger.debug(f"Batch Ranges length: {len(batches)}")

    # Web Mercator to EPSG:3857 with custom y_0=27445 to match frontend config
    transformer = Transformer.from_crs(
        "+proj=longlat +datum=WGS84 +no_defs",
        "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=27445 +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    )
    sw_x, sw_y = transformer.transform(
        rectangle_bound.south_west.lng, rectangle_bound.south_west.lat
    )
    ne_x, ne_y = transformer.transform(
        rectangle_bound.north_east.lng, rectangle_bound.north_east.lat
    )

    logger.debug(f"SW: {sw_x}, {sw_y}")
    logger.debug(f"NE: {ne_x}, {ne_y}")

    bbox = shapely.box(sw_x, sw_y, ne_x, ne_y)
    bbox_bounds = bbox.bounds  # (minx, miny, maxx, maxy) for quick checks

    filtered_agents = []
    

    if len(batches) > 1:
        # Use parallel processing for multiple batches
        logger.info(
            f"Using parallel processing for filtering by bounds with multiple batches and cores: {num_cores}"
        )
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_cores,
                mp_context=mp.get_context("spawn"),  # Use spawn for better isolation
            ) as executor:
                batch_data = []
                for start, end in batches:
                    batch_centroids = all_centroids[start:end]
                    batch_data.append((start, sw_x, sw_y, ne_x, ne_y, batch_centroids))

                # Submit all jobs
                future_to_batch = {
                    executor.submit(_filter_agents_batch, batch_info): batch_info
                    for batch_info in batch_data
                }

                # Collect results with timeout and error handling
                for future in concurrent.futures.as_completed(
                    future_to_batch, timeout=300
                ):
                    batch_info = future_to_batch[future]
                    try:
                        result_ids = future.result()
                        if result_ids is not None:
                            
                            for idx in result_ids:
                                filtered_agents.append(process_agent_for_rendering(agents[idx], should_show_polygons))

                        else:
                            logger.warning(
                                f"Batch {batch_info[0]}-{batch_info[1]} returned None"
                            )
                    except Exception as e:
                        logger.error(
                            f"Batch {batch_info[0]}-{batch_info[1]} failed: {str(e)}"
                        )
                        # Continue processing other batches
                        continue

        except Exception as e:
            logger.error(f"ProcessPoolExecutor failed: {str(e)}")
            logger.info(f"Falling back to serial processing with quick bbox check")
            # candidates = [
            #     agent for agent in agents if quick_bbox_check(agent, bbox_bounds)
            # ]
            contained_agents = [
                agent for agent in agents if bbox.contains(agent.centroid)
            ]
            processed_agents = preprocess_agents_for_rendering(contained_agents)
            filtered_agents.extend(processed_agents)
    else:
        logger.info(
            f"Using serial processing for filtering by bounds with single batch"
        )

        # candidates = [agent for agent in agents if quick_bbox_check(agent, bbox_bounds)]
        contained_agents = [
            agent for agent in agents if bbox.contains(agent.centroid)
        ]
        processed_agents = preprocess_agents_for_rendering(contained_agents)
        filtered_agents.extend(processed_agents)

    # Clean up
    del all_centroids
    gc.collect()

    logger.info(
        f"Filtered to {len(filtered_agents)} agents from {len(agents)} original agents"
    )
    

    if len(filtered_agents) == 0:
        logger.info(f"No agents found within bounds, returning empty list")
        return []

    if len(filtered_agents) > max_elements:
        logger.info(f"agents after bounds filtering {len(filtered_agents)} > max_elements {max_elements}, sampling {max_elements} random indices")
        random_indices = random.sample(range(len(filtered_agents)), max_elements)
        return [filtered_agents[i] for i in random_indices]

    return filtered_agents


def filter_by_bounds(agents, rectangle_bound: RectangleBound):

    bounds = rectangle_bound

    # Web Mercator to EPSG:3857 with custom y_0=27445 to match frontend config
    transformer = Transformer.from_crs(
        "+proj=longlat +datum=WGS84 +no_defs",
        "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=27445 +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    )

    if bounds is not None:
        # Transform bbox corners from lat/lng to Web Mercator (EPSG:3857) with custom y_0
        sw_x, sw_y = transformer.transform(bounds.south_west.lng, bounds.south_west.lat)
        ne_x, ne_y = transformer.transform(bounds.north_east.lng, bounds.north_east.lat)

        logger.debug(f"SW: {sw_x}, {sw_y}")
        logger.debug(f"NE: {ne_x}, {ne_y}")

        bbox = shapely.box(sw_x, sw_y, ne_x, ne_y)

        logger.debug(f"BBOX: {bbox}")
        logger.debug(f"Agents[0] centroid: {agents[0].centroid}")
        # Check if centroid is inside the bounds (bbox)
        filtered_agents = [a for a in agents if bbox.contains(a.centroid)]

    return filtered_agents
