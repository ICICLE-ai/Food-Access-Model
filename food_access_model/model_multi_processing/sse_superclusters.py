from typing import Tuple
import shapely
import gc
import concurrent.futures
import multiprocessing as mp
import traceback

from food_access_model.api.models import DrawableAgent, GetDrawableAgentsRequest, MAX_SUPERCLUSTER_POINTS
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

async def stream_superclusters(request: GetDrawableAgentsRequest, repository: DBRepository) -> list[DrawableAgent]:

    logger.info(f"Processing superclusters")
    
    if repository is None:
        raise Exception("Repository not initialized")

    bounds = request.bounds
    
    cached_supercluster_agents = simple_cache.get(f"supercluster_agents")
    if cached_supercluster_agents:
        async for item in cached_supercluster_agents:
            yield item

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
    num_cores = min(mp.cpu_count(), MAX_CPUS)  # Don't use all CPUs
    batch_size = max(1000, len(agents_list) // (num_cores * 2))  # Ensure reasonable batch sizes
    chunk_size = batch_size

    logger.debug(f"Using {num_cores} cores for supercluster processing")
    logger.debug(f"Using batch size: {batch_size}")

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

    max_workers = min(mp.cpu_count(), MAX_CPUS)  # TODO:Don't use all CPUs and finetune this to improve performance
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
                            yield result
                        else:
                            logger.warning(f"Batch {batch_info[0]}-{batch_info[1]} returned None")
                    except Exception as e:
                        logger.error(f"Batch {batch_info[0]}-{batch_info[1]} failed: {str(e)}")
                        # Continue processing other batches
                        continue
                        
        except Exception as e:
            logger.error(f"ProcessPoolExecutor failed: {str(e)}")
            traceback.print_exc()
            # Fallback to serial processing
            logger.info("Falling back to serial processing")
            async for item in process_superclusters_serial_fallback(all_centroids, batch_ranges):
                yield item
    else:
        # For single batch, call directly
        start, end = batch_ranges[0]
        batch_centroids = all_centroids[start:end]
        result = _process_supercluster_batch_safe((start, end, batch_centroids))
        if result is not None:
            yield result

    # Clean up
    del all_centroids
    gc.collect()

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
