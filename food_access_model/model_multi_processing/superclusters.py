from typing import Tuple
import shapely
import gc
import concurrent.futures
import multiprocessing as mp
import traceback
import time

from food_access_model.api.models import DrawableAgent, GetDrawableAgentsRequest, MAX_SUPERCLUSTER_POINTS, RectangleBound, Bound
from food_access_model.api.configs import MAX_CPUS
from food_access_model.repository.db_repository import DBRepository
from food_access_model.utils.cache import get_cache

import logging

logger = logging.getLogger(__name__)

simple_cache = get_cache()


def get_supercluster_xy(centroids) -> Tuple[float, float]:
    centroid = shapely.MultiPoint(centroids).centroid
    return centroid.x, centroid.y

def get_agent_centroid(agent) -> Tuple[float, float]:
    return (agent.centroid.x, agent.centroid.y)

# Move this function to module level so it can be pickled
def _process_supercluster_batch_safe(batch_info):
    """Process a batch of centroids into a supercluster agent with error handling."""
    try:
        idx, agent_centroids = batch_info
        
        # Process only the centroids we need
        supercluster_xy = get_supercluster_xy(agent_centroids)
        
        result = DrawableAgent(
            id=f"supercluster_{idx}",
            agent_type="supercluster",
            centroid_xy=supercluster_xy,
            properties={
                "ID": f"supercluster_{idx}",
                "Size": len(agent_centroids),
                "Index": idx,
            },
        )
        
        # Force garbage collection to free memory
        gc.collect()
        return result
        
    except Exception as e:
        logger.error(f"Error processing batch {start_idx}-{end_idx}: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a placeholder or None to indicate failure
        return None

def _extract_centroids_batch(agents_slice):
    """Extract centroids from a slice of agents - separate function for memory efficiency."""
    return [get_agent_centroid(agent) for agent in agents_slice]

def process_superclusters(request: GetDrawableAgentsRequest, repository: DBRepository) -> list[DrawableAgent]:

    logger.info(f"Processing superclusters")
    
    if repository is None:
        raise Exception("Repository not initialized")

    bounds = request.bounds
    
    cached_supercluster_agents = simple_cache.get(f"supercluster_agents")
    if cached_supercluster_agents:
        logger.info("Using cached supercluster agents")
        return cached_supercluster_agents

    # Calculate batch size
    model = repository.get_model()
    agents_list = model.space.agents
    
    # Strategy 1: Use smaller batch sizes to reduce memory pressure
    batch_size = len(agents_list) // MAX_SUPERCLUSTER_POINTS
    batch_size = max(1000, min(5000, batch_size))  # Smaller batches for stability
    
    logger.debug(f"Agents list length: {len(agents_list)}")
    logger.debug(f"Batch Size: {batch_size}")

    logger.debug("Pre-extracting centroids...")
    
    # Process centroids in chunks to avoid memory issues
    num_cores = min(mp.cpu_count(), MAX_CPUS)  # Don't use all CPUs. TODO: In Darwin we have 64 but via vscode are allowed to use just 4
    # num_cores = mp.cpu_count() - 1  # Don't use all CPUs
    batch_size = max(1000, len(agents_list) // (num_cores * 2))  # Ensure reasonable batch sizes
    chunk_size = batch_size

    logger.info(f"Using {num_cores} cores for supercluster processing")
    logger.info(f"Using batch size: {batch_size}")

    all_centroids = []
    for i in range(0, len(agents_list), chunk_size):
        chunk = agents_list[i:i + chunk_size]
        chunk_centroids = [get_agent_centroid(agent) for agent in chunk]
        all_centroids.extend(chunk_centroids)
        
        #TODO: Force garbage collection after each chunk. CHeck safeness
        del chunk
        gc.collect()
    
    batch_ranges = [
        (i, min(i + batch_size, len(all_centroids)))
        for i in range(0, len(all_centroids), batch_size)
    ]
    
    logger.debug(f"Batch Ranges length: {len(batch_ranges)}")

    supercluster_agents = []
    
    # max_workers = min(mp.cpu_count(), 4)  # TODO:Don't use all CPUs and finetune this to improve performance
    max_workers = num_cores  # TODO:Don't use all CPUs and finetune this to improve performance
    logger.info(f"Starting supercluster processing in parallel with {max_workers} workers")
    
    if len(batch_ranges) > 1:
        
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context('spawn')  # Use spawn for better isolation
            ) as executor:
                # Prepare batch data with actual centroids
                batch_data = []
                for start, end in batch_ranges:
                    batch_centroids = all_centroids[start:end]
                    batch_data.append((start, batch_centroids))
                
                # Submit all jobs
                future_to_batch = {
                    executor.submit(_process_supercluster_batch_safe, batch_info): batch_info
                    for batch_info in batch_data
                }
                
                # Collect results with timeout and error handling
                for future in concurrent.futures.as_completed(future_to_batch, timeout=300):
                    batch_info = future_to_batch[future]
                    try:
                        result = future.result()
                        if result is not None:
                            supercluster_agents.append(result)
                        else:
                            logger.warning(f"Batch {batch_info[0]} returned None")
                    except Exception as e:
                        logger.error(f"Batch {batch_info[0]} failed: {str(e)}")
                        # Continue processing other batches
                        continue
                        
        except Exception as e:
            logger.error(f"ProcessPoolExecutor failed: {str(e)}")
            traceback.print_exc()
            # Fallback to serial processing
            logger.info("Falling back to serial processing")
            return process_superclusters_serial_fallback(all_centroids, batch_ranges)
    else:
        # For single batch, call directly
        start, end = batch_ranges[0]
        batch_centroids = all_centroids[start:end]
        result = _process_supercluster_batch_safe((start, end, batch_centroids))
        if result is not None:
            supercluster_agents = [result]

    # Clean up
    del all_centroids
    gc.collect()
    
    # Filter out None results
    supercluster_agents = [agent for agent in supercluster_agents if agent is not None]
    
    logger.info(f"Successfully processed {len(supercluster_agents)} superclusters")
    
    simple_cache.set(f"supercluster_agents", supercluster_agents)
    return supercluster_agents

def process_superclusters_serial_fallback(all_centroids, batch_ranges):
    """Fallback to serial processing if multiprocessing fails."""
    logger.debug("Processing superclusters serially as fallback")
    supercluster_agents = []
    
    for start, end in batch_ranges:
        try:
            batch_centroids = all_centroids[start:end]
            result = _process_supercluster_batch_safe((start, end, batch_centroids))
            if result is not None:
                supercluster_agents.append(result)
        except Exception as e:
            logger.error(f"Serial processing failed for batch {start}-{end}: {str(e)}")
            continue
    
    return supercluster_agents



INITIAL_NORTH_EAST = Bound(lat=40.04077889930881, lng=-82.93544769287111)
INITIAL_SOUTH_WEST = Bound(lat=39.89790562479971, lng=-83.07689666748048)

INITIAL_CENTER = (39.938806, -82.972361)
INITIAL_ZOOM = 13;

def initialize_supercluster_cache(repository: DBRepository):
    """Initialize the supercluster cache."""
    logger.info("Initializing Geometry Supercluster cache")
    start_time = time.time()
    
    model = repository.get_model()
    agents_list = model.space.agents
    drawable_request = GetDrawableAgentsRequest(
        bounds=RectangleBound(north_east=INITIAL_NORTH_EAST, south_west=INITIAL_SOUTH_WEST),
        step_number=0,
        max_elements=100,
        center=INITIAL_CENTER,
        zoom=INITIAL_ZOOM
    )
    supercluster_agents = process_superclusters(drawable_request, repository)
    end_time = time.time()
    
    logger.info(f"Geometry Supercluster cache initialized in {end_time - start_time} seconds")
    return supercluster_agents