"""
Copyright Notice: This code is part of the mesa-geo project, which is licensed under the Apache 2.0 License.
https://github.com/projectmesa/mesa-geo?tab=Apache-2.0-1-ov-file.

The code below is a modification for an specific use case and is not part of the original mesa-geo project.
This code is a modified version of the batchrunner.py file from the mesa-geo project.
This code is used to run a batch of simulations in parallel using the multiprocessing library.

Changes:
This code changes the way parameters are passed to the model. 
"""

from decimal import Decimal
from functools import partial
from multiprocessing import Pool, active_children


#for p in active_children():
#    p.terminate()
#    p.join()

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from mesa import Agent
from tqdm.auto import tqdm

from mesa.model import Model
import os

def batch_run(
    model_cls: Type[Model],
    parameters: Mapping[str, Union[Any, Iterable[Any]]],
    number_processes: Optional[int] = None,
    data_collection_period: int = -1,
    max_steps: int = 1000,
    display_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Inspired from the Mesa-GEO batch_run function.
    Batch run a mesa model with a set of parameter values.

    Parameters
    ----------
    model_cls : Type[Model]
        The model class to batch-run
    parameters : Mapping[str, Union[Any, Iterable[Any]]],
        Dictionary with model parameters over which to run the model. You can either pass single values or iterables.
        For the purpose of FASS this MUST contain a "households" and a "stores" key, both storing lists of data.
    number_processes : int, optional
        Number of processes used, by default 1. Set this to None if you want to use all CPUs.
    iterations : int, optional
        Number of iterations for each parameter combination, by default 1
    data_collection_period : int, optional
        Number of steps after which data gets collected, by default -1 (end of episode)
    max_steps : int, optional
        Maximum number of model steps after which the model halts, by default 1000
    display_progress : bool, optional
        Display batch run process, by default True

    Returns
    -------
    List[Dict[str, Any]]
        [description]
    """

    runs_list = []
    run_id = 0

    print(f"Batch runner began", flush=True)
     
    household_chunks = create_household_chunks(parameters["households"], number_processes)
    
    # for iteration in range(iterations):
        
    for household_group in household_chunks:
        model_config = {"stores": parameters['stores'], "households": household_group}
        runs_list.append((run_id, 0, model_config))
        run_id += 1
        

    process_func = partial(
        _model_run_func,
        model_cls,
        max_steps=max_steps,
        data_collection_period=data_collection_period,
    )

    
    results: List[Dict[str, Any]] = []

    with tqdm(total=len(runs_list), disable=not display_progress) as pbar:
        if number_processes == 1:
            for run in runs_list:
                data = process_func(run)
                results.append(data)
                pbar.update()
               
        else:
            try:
                with Pool(number_processes) as pool:
                    for data in pool.imap_unordered(process_func, runs_list):
                        results.append(data)
                        pbar.update()
                    # Ensure all processes are properly terminated
                    pool.close()
                    pool.join()
            except Exception as e:
                return {"error": str(e), "input": data}
                
    print(f"tqdm FUNCTION END", flush=True)
    return results

def create_household_chunks(households, number_processes : Optional[int] = None):
    
    if number_processes is None:
        number_processes = os.cpu_count() or 1
    
    # Divide households into groups based on the number of cores
    household_groups = []
    chunk_size = len(households) // number_processes

    last_chunk = 0

    for i in range(0, number_processes):
        household_groups.append(households[last_chunk:last_chunk + chunk_size])
        last_chunk += chunk_size

    if last_chunk < len(households):
        household_groups[-1].extend(households[last_chunk:])
    
    return household_groups
    
def create_store_records(stores: List[Agent]):
    store_records = []
    for store in stores:
        name = store.name
        type = store.type
        geometry = store.raw_geometry
        store = [type, geometry, name]

        store_records.append(store)
    return store_records

def extract_decimal(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return value

def create_household_records(households: List[Dict[str, Any]]):
    household_records = []
    for household in households:
        id = extract_decimal(household["AgentID"])
        polygon = household["Geometry"]
        income = extract_decimal(household["Income"])
        household_size = extract_decimal(household["Household Size"])
        vehicles = extract_decimal(household["Vehicles"])
        number_of_workers = extract_decimal(household["Number of Workers"])
        walking_time = extract_decimal(household["Walking time"])
        biking_time = extract_decimal(household["Biking time"])
        transit_time = extract_decimal(household["Transit time"])
        driving_time = extract_decimal(household["Driving time"])
        distance_to_closest_store = extract_decimal(household["Closest Store (Miles)"])
        num_store_within_mile = extract_decimal(household["Stores within 1 Mile"])
        mfai = extract_decimal(household["Food Access Score"])
        color = household["Color"]
        household = [id, polygon, income, household_size, vehicles, number_of_workers, walking_time, biking_time, transit_time, driving_time, distance_to_closest_store, num_store_within_mile, mfai, color]
        household_records.append(household)

    return household_records

def _model_run_func(
    model_cls: Type[Model],
    run: Tuple[int, int, Dict[str, Any]],
    max_steps: int,
    data_collection_period: int,
) -> List[Dict[str, Any]]:
    """Run a single model run and collect model and agent data.

    Parameters
    ----------
    model_cls : Type[Model]
        The model class to batch-run
    run: Tuple[int, int, Dict[str, Any]]
        The run id, iteration number, and kwargs for this run
    max_steps : int
        Maximum number of model steps after which the model halts, by default 1000
    data_collection_period : int
        Number of steps after which data gets collected

    Returns
    -------
    List[Dict[str, Any]]
        Return model_data, agent_data from the reporters
    """

    run_id, iteration, kwargs = run

    model = model_cls(**kwargs)
    """WARNING: Original mesa-geo implementation has a <= max_steps comparison.
    I changed from <= to < as a hotfix to avoid running the model for an extra step.
    This is a temporary fix and should be reviewed in the future.
    """
    while model.running and model.schedule.steps < max_steps:
        model.step()

    
    data = []
    
    steps = list(range(1, model.schedule.steps, data_collection_period))

    if not steps or steps[-1] != model.schedule.steps - 1:
        steps.append(model.schedule.steps - 1)

    store_records = create_store_records(model.stores_list)
    for step in steps:
        model_data, all_agents_data = _collect_data(model, step)

        household_records = create_household_records(all_agents_data)

        # If there are agent_reporters, then create an entry for each agent
        if all_agents_data:
            stepdata = [
                {
                    "RunId": run_id,
                    "iteration": iteration,
                    "Step": step,
                    "kwargs": kwargs,
                    "model_data": model_data,
                    #TODO: Only housholds are being added to schedule when creating the model. That's why the data collector only process households.
                    "households": household_records,
                    "stores": store_records
                }
            ]
        # If there is only model data, then create a single entry for the step
        else:
            stepdata = [
                {
                    "RunId": run_id,
                    "iteration": iteration,
                    "Step": step,
                    "kwargs": kwargs,
                    "model_data": model_data,
                    "raw_stores": model.stores,
                    "raw_households": model.households
                }
            ]
        data.extend(stepdata)


    print("Finished data collection", flush=True)
    return data

def _collect_data(
    model: Model,
    step: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Collect model and agent data from a model using mesas datacollector."""
    dc = model.datacollector

    model_data = {param: values[step] for param, values in dc.model_vars.items()}

    all_agents_data = []
    """
    Note: In the original code the implementation accesses _agent_records[step].
    However, when debugging this code and after reading the mesa-geo code, I realized that the data is stored in a different way.
    The data is stored in the _agent_records dictionary with steps starting from 1.
    So, I am adding 1 to step to access the correct data.
    """
    raw_agent_data = dc._agent_records.get(step+1, [])
    for data in raw_agent_data:
        agent_dict = {"AgentID": data[1]}
        agent_dict.update(zip(dc.agent_reporters, data[2:]))
        all_agents_data.append(agent_dict)

    return model_data, all_agents_data
