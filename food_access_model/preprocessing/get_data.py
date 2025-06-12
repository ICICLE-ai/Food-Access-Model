import osmnx as ox
import psycopg2
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
from rtree.index import Index as RTreeIndex
from psycopg2.extensions import connection as Connection, cursor as Cursor
import rtree
import math
import pandas as pd
import requests
from zipfile import ZipFile
import tempfile
import shapely
#import googlemaps
from datetime import datetime
import os
import numpy as np
import geopandas
import random
from psycopg2 import extras
import shapely.geometry as geometry
from pyproj import Transformer
from typing import List, Tuple, Optional, Callable, Any
from io import BytesIO
import logging
from household_constants import(
    YEAR, 
    PLACE_NAME, 
    CENTER_POINT, 
    DIST, 
    FIPSCODE, 
    households_variables_dict,
    households_key_list,
    household_values_list, 
    income_ranges,
    size_index_dict,
    workers_index_dict
)  

COUNTY_CODE = FIPSCODE[2:]                  
STATE_CODE = FIPSCODE[:2]   

PASS = os.getenv("DB_PASS")
if not PASS:
    logging.critical("DB_PASS environment variable not set.")
    raise ValueError("DB_PASS is required.")     
APIKEY = os.getenv("APIKEY")
if not APIKEY:
    logging.critical("APIKEY environment variable not set.")
    raise ValueError("APIKEY is required.")   
USER = os.getenv("DB_USER")
if not USER:
    logging.critical("DB_USER environment variable not set.")
    raise ValueError("DB_USER is required.") 
NAME = os.getenv("DB_NAME")
if not NAME:
    logging.critical("DB_NAME environment variable not set.")
    raise ValueError("DB_NAME is required.") 
HOST = os.getenv("DB_HOST")
if not HOST:
    logging.critical("DB_HOST environment variable not set.")
    raise ValueError("DB_HOST is required.") 
PORT = os.getenv("DB_PORT")
if not PORT:
    logging.critical("DB_PORT environment variable not set.")
    raise ValueError("DB_PORT is required.") 

def fetch_county_data(
        household_keys: List[str], 
        year: str, state_code: str, 
        county_code: str, 
        api_key: str) -> pd.DataFrame:
    """
    Fetch household data from the US Census API for a given county and state.

    Args:
        household_keys (List[str]): List of household variable keys.
        year (str): The year for the dataset (e.g., '2021').
        state_code (str): Two-digit FIPS state code.
        county_code (str): Three-digit FIPS county code.
        api_key (str): API key for the US Census API.

    Returns:
        pd.DataFrame: Merged DataFrame containing all retrieved household data.
    """
    if not household_keys or not isinstance(household_keys, list):
        logging.error("household_keys must be a non-empty list.")
        return pd.DataFrame()
    if not year or not isinstance(year, str):
        logging.error("year must be a non-empty string.")
        return pd.DataFrame()
    if not state_code or not isinstance(state_code, str) or len(state_code) != 2:
        logging.error("state_code must be a 2-digit string.")
        return pd.DataFrame()
    if not county_code or not isinstance(county_code, str) or len(county_code) != 3:
        logging.error("county_code must be a 3-digit string.")
        return pd.DataFrame()
    if not api_key or not isinstance(api_key, str):
        logging.error("api_key must be provided and a string.")
        return pd.DataFrame()
        
    county_data = pd.DataFrame()

    for count in range((len(household_keys) // 50) + 1):
        if (count + 1) * 50 > len(household_keys):
            variables = ",".join(household_keys[(50 * count):])
        elif count == 0:
            if ((len(household_keys) // 50) + 1) == 1:
                variables = ",".join(household_keys[:])
            else:
                variables = ",".join(household_keys[:(50 * (count + 1) - 1)])
        else:
            variables = ",".join(household_keys[(50 * count):(50 * (count + 1) - 1)])

        url = (
            f"https://api.census.gov/data/{year}/acs/acs5?"
            f"get=NAME,{variables}&for=tract:*&in=state:{state_code}&in=county:{county_code}&key={api_key}"
        )
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as e:
                logging.error(f"JSON decode error for Census API: {e} | URL: {url}")
                continue
            # checks for header and data rows
            if not data or len(data) < 2:
                logging.error(f"Empty or malformed Census API response: {url}")
                continue
        except requests.RequestException as e:
            logging.error(f"HTTP request failed for Census API: {e} | URL: {url}")
            continue

        if not county_data.empty:
            county_data = pd.merge(
                pd.DataFrame(data[1:], columns=data[0]),
                county_data,
                on='NAME',
                how='inner'
            )
        else:
            county_data = pd.DataFrame(data[1:], columns= data[0])
    return county_data

# Initialize county data using the function
COUNTY_DATA = fetch_county_data(households_key_list, YEAR, STATE_CODE, COUNTY_CODE, APIKEY)


def load_and_merge_geodata(
    year: str,
    state_code: str,
    county_code: str,
    census_data: pd.DataFrame
) -> geopandas.GeoDataFrame:
    """
    Download and merge census tract geometries with census data for a specific county.

    Args:
        year (str): Year of the census data.
        state_code (str): Two-digit FIPS code of the state.
        county_code (str): Three-digit FIPS code of the county.
        census_data (pd.DataFrame): Household census data.

    Returns:
        geopandas.GeoDataFrame: Merged geospatial dataframe for the specified county.
    """
    if not year or not isinstance(year, str):
        logging.error("year must be a non-empty string.")
        return None
    if not state_code or not isinstance(state_code, str) or len(state_code) != 2:
        logging.error("state_code must be a 2-digit string.")
        return None
    if not county_code or not isinstance(county_code, str) or len(county_code) != 3:
        logging.error("county_code must be a 3-digit string.")
        return None
    if census_data is None or not isinstance(census_data, pd.DataFrame) or census_data.empty:
        logging.error("census_data must be a non-empty DataFrame.")
        return None
        
    tract_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_code}_tract.zip"

    try:
        response = requests.get(tract_url, timeout=30)
        response.raise_for_status()
        zip_bytes = BytesIO(response.content)
    except requests.RequestException as e:
        logging.error(f"HTTP request failed for shapefile: {e} | URL: {tract_url}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error downloading shapefile: {e} | URL: {tract_url}")
        return None

    with ZipFile(zip_bytes) as zip_ref:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_ref.extractall(tmpdirname)

            for root, dirs, files in os.walk(tmpdirname):
                for file in files:
                    if file.endswith(".shp") or file.endswith(".geojson"):
                        file_path = os.path.join(root, file)
                        geodata = geopandas.read_file(file_path).to_crs("epsg:3857")

    county_geodata = geodata[geodata['COUNTYFP'] == county_code].copy()
    county_geodata = county_geodata.rename(columns={"TRACTCE": "tract_y"})
    county_geodata["tract_y"] = county_geodata["tract_y"].astype(int)
    census_data["tract_y"] = census_data["tract_y"].astype(int)

    try:
         merged_data = pd.merge(county_geodata, census_data, on="tract_y", how="inner")
    except Exception as e:
        logging.error(f"Failed to merge DataFrames: {e}")
        return None
    merged_data.rename(columns=households_variables_dict, inplace=True)
    merged_data = merged_data.to_crs("epsg:3857")

    return merged_data

# Run and create merged geospatial data
data = load_and_merge_geodata(YEAR, STATE_CODE, COUNTY_CODE, COUNTY_DATA)

# Create spatial index from geometry
tract_index = STRtree(data["geometry"])


def initialize_database_tables(
    host: str,
    database: str,
    user: str,
    password: str,
    port: str
) -> Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """
    Connect to the PostgreSQL database and initialize required tables.

    Args:
        host (str): Database host address.
        database (str): Name of the database.
        user (str): Database username.
        password (str): User's database password.
        port (str): Port number for the database connection.

    Returns:
        Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]: 
            A tuple containing the active connection and cursor objects.
    """
    try:
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        cursor = connection.cursor()
    except psycopg2.Error as e:
        logging.error(f"Failed to connect to the database: {e}")
        return None, None
    

    # Drop tables if they already exist
    try:
        cursor.execute('DROP TABLE IF EXISTS roads;')
        cursor.execute('DROP TABLE IF EXISTS food_stores;')
    except psycopg2.Error as e:
        logging.error(f"Database table operation failed: {e}")
        connection.rollback()
        return None, None

    # Create tables
    create_roads_query = '''
    CREATE TABLE roads (
        name TEXT,
        highway VARCHAR(30),
        length NUMERIC,
        geometry TEXT,
        service VARCHAR(30)
    );
    '''
    create_food_stores_query = '''
    CREATE TABLE food_stores (
        shop VARCHAR(15),
        geometry TEXT,
        name VARCHAR(50)
    );
    '''
    cursor.execute(create_roads_query)
    cursor.execute(create_food_stores_query)

    return connection, cursor

# Call the function to set up the database and get cursor
connection, cursor = initialize_database_tables(HOST, NAME, USER, PASS, PORT)


def process_road_network(
    center_point: Tuple[float, float],
    dist: float
) -> Tuple[List[geometry.base.BaseGeometry], List[geometry.base.BaseGeometry], List[Tuple]]:
    """
    Process road network from OpenStreetMap and prepare geometry buffers and SQL-ready tuples.

    Args:
        center_point (Tuple[float, float]): Latitude and longitude of the center location.
        dist (float): Distance in meters to define the radius of the map from the center point.

    Returns:
        Tuple[
            List[geometry.base.BaseGeometry], 
            List[geometry.base.BaseGeometry], 
            List[Tuple]
        ]: A tuple containing:
            - List of buffered geometries for map rendering
            - List of buffered geometries for housing area estimation
            - List of road attribute tuples ready for SQL insertion
    """
    map_elements: List[geometry.base.BaseGeometry] = []
    housing_areas: List[geometry.base.BaseGeometry] = []

    G = ox.graph_from_point(center_point, dist=dist, network_type='all', retain_all=True)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_edges = gdf_edges.to_crs("epsg:3857")

    if "service" in gdf_edges.columns:
        gdf_edges = gdf_edges[["name", "highway", "length", "geometry", "service"]]
    else:
        gdf_edges["service"] = None  # ensure consistent schema
        gdf_edges = gdf_edges[["name", "highway", "length", "geometry", "service"]]

    for _, row in gdf_edges.iterrows():
        if row["highway"] in ["residential", "living_street"]:
            housing_areas.append(row["geometry"].buffer(30))
            map_elements.append(row["geometry"].buffer(3))
        elif row["service"] == "alley":
            housing_areas.append(row["geometry"].buffer(30))
            map_elements.append(row["geometry"].buffer(3))
        elif row["highway"] == "motorway":
            map_elements.append(row["geometry"].buffer(100))
        elif row["highway"] == "trunk":
            map_elements.append(row["geometry"].buffer(30))
        elif row["highway"] == "primary":
            map_elements.append(row["geometry"].buffer(10))
        elif row["highway"] == "secondary":
            map_elements.append(row["geometry"].buffer(10))
        elif isinstance(row["geometry"], LineString):
            map_elements.append(row["geometry"])

    gdf_edges["length"] = gdf_edges["length"].astype(int)
    gdf_edges["geometry"] = gdf_edges["geometry"].astype(str)

    data_tuples = list(gdf_edges.itertuples(index=False, name=None))

    return map_elements, housing_areas, data_tuples


# Run the road network processor
map_elements, housing_areas, data_tuples = process_road_network(CENTER_POINT, DIST)

# Prepare the SQL insertion query
roads_query = "INSERT INTO roads (name, highway, length, geometry, service) VALUES %s"


def process_food_stores(
    center_point: Tuple[float, float],
    dist: float,
    map_elements: List[geometry.base.BaseGeometry],
    cursor: psycopg2.extensions.cursor
) -> STRtree:
    """
    Retrieve and process food store locations from OpenStreetMap, convert them into geometric shapes,
    and insert them into the database.

    Args:
        center_point (Tuple[float, float]): Latitude and longitude for the area of interest.
        dist (float): Distance in meters for the search radius (3x for food stores).
        map_elements (List[BaseGeometry]): List to append buffered polygons representing stores.
        cursor (psycopg2.extensions.cursor): Cursor for executing database insert queries.

    Returns:
        STRtree: Spatial index of all geometric elements including food stores.
    """
    features = ox.features.features_from_point(
        center_point,
        dist=dist * 3,
        tags={"shop": [
            "convenience", "supermarket", "butcher", "wholesale",
            "farm", "greengrocer", "health_food", "grocery"
        ]}
    )

    features = features.to_crs("epsg:3857")
    features = features[["shop", "geometry", "name"]]

    store_tuples: List[Tuple[str, str, str]] = []
    food_stores_query = "INSERT INTO food_stores (shop, geometry, name) VALUES %s"

    for _, row in features.iterrows():
        point = row["geometry"].centroid if not isinstance(row["geometry"], Point) else row["geometry"]

        if row["shop"] in ["supermarket", "grocery", "greengrocer"]:
            polygon = Polygon([
                (point.x + 50 * math.cos(math.radians(angle)), point.y + 50 * math.sin(math.radians(angle)))
                for angle in range(0, 360, 60)
            ])
        else:
            polygon = Polygon([
                (point.x, point.y + 20),
                (point.x + 25, point.y - 30),
                (point.x - 25, point.y - 30)
            ])

        map_elements.append(polygon.buffer(20))
        store_tuples.append((str(row["shop"]), str(polygon), str(row["name"])))
    try:
        extras.execute_values(cursor, food_stores_query, store_tuples)
    except psycopg2.Error as e:
        logging.error(f"Databse insertion failed: {e}")
        connection.rollback()
        raise

    return STRtree(map_elements)


# Call the function and get the spatial index of mapped elements
map_elements_index = process_food_stores(CENTER_POINT, DIST, map_elements, cursor)


def create_households_table(cursor: psycopg2.extensions.cursor) -> str:
    """
    Create the 'households' table in the database after dropping it if it already exists.

    Args:
        cursor (psycopg2.extensions.cursor): Active database cursor.

    Returns:
        str: Prepared SQL insert query string for inserting household data.
    """
    create_households_query = '''
    CREATE TABLE households (
        id NUMERIC,
        polygon TEXT,
        income NUMERIC,
        household_size NUMERIC,
        vehicles NUMERIC,
        number_of_workers NUMERIC,
        walking_time TEXT,
        biking_time TEXT,
        transit_time TEXT,
        driving_time TEXT
    );
    '''

    cursor.execute('DROP TABLE IF EXISTS households;')
    cursor.execute(create_households_query)

    household_query = """
    INSERT INTO households 
    (id,
     polygon,
     income,
     household_size,
     vehicles,
     number_of_workers,
     walking_time,
     biking_time,
     transit_time,
     driving_time) 
    VALUES %s
    """
    return household_query


def close_db_connection(connection: psycopg2.extensions.connection, cursor: psycopg2.extensions.cursor) -> None:
    """
    Commit any pending transactions and close the database connection and cursor.

    Args:
        connection (psycopg2.extensions.connection): The active database connection.
        cursor (psycopg2.extensions.cursor): The active database cursor.
    """
    connection.commit()
    cursor.close()
    connection.close()


def create_house_polygon(location: Point) -> Polygon:
    """
    Create a synthetic house polygon at a given location.

    Args:
        location (Point): Center point of the house.

    Returns:
        Polygon: A polygon representing the house footprint.
    """
    return Polygon([
        (location.x + 10, location.y + 10),
        (location.x, location.y + 20),
        (location.x - 10, location.y + 10),
        (location.x - 10, location.y - 5),
        (location.x - 3, location.y - 5),
        (location.x - 3, location.y + 3),
        (location.x + 3, location.y + 3),
        (location.x + 3, location.y - 5),
        (location.x + 10, location.y - 5),
        (location.x + 10, location.y + 10)  
    ])



def place_houses_in_area(housing_area: Polygon, spacing: float = 30.0) -> List[Polygon]:
    """
    Place house-shaped polygons around the edges of a housing area polygon.

    Args:
        housing_area (Polygon): A polygon representing the buffer around a residential road.
        spacing (float): Distance between each house along the edge.

    Returns:
        List[Polygon]: List of synthetic house polygons.
    """
    houses: List[Polygon] = []
    exterior_coords = list(housing_area.exterior.coords)
    edges = [LineString([exterior_coords[i], exterior_coords[i + 1]])
             for i in range(len(exterior_coords) - 1)]

    for edge in edges:
        length = edge.length
        coord1, coord2 = edge.coords[0], edge.coords[1]
        vector_direction = (coord2[0] - coord1[0], coord2[1] - coord1[1])
        magnitude = (vector_direction[0] ** 2 + vector_direction[1] ** 2) ** 0.5

        if magnitude == 0:
            continue

        norm_vector = (vector_direction[0] / magnitude, vector_direction[1] / magnitude)

        for i in range(int(magnitude // spacing) + 1):
            location = Point(coord1[0] + norm_vector[0] * i * spacing,
                             coord1[1] + norm_vector[1] * i * spacing)

            house = create_house_polygon(location)

            houses.append(house)

    return houses

def is_valid_house(
    house: Polygon,
    map_elements_index: STRtree,
    houses_index: rtree.index.Index,
    map_elements: List[Polygon]
) -> bool:
    """
    Check whether a house polygon is valid by verifying it does not intersect
    with roads or existing houses.

    Args:
        house (Polygon): The house polygon to check.
        map_elements_index (STRtree): Spatial index for roads and store areas.
        houses_index (rtree.index.Index): R-tree index of all placed houses.
        map_elements (List[Polygon]): The list of buffered roads and stores.

    Returns:
        bool: True if house is valid; False if it intersects with other features.
    """
    # Check collision with road/store polygons
    search_area = Polygon([
        (house.centroid.x + 10, house.centroid.y + 20),
        (house.centroid.x - 10, house.centroid.y + 20),
        (house.centroid.x - 10, house.centroid.y - 5),
        (house.centroid.x + 10, house.centroid.y - 5)
    ])

    for index in map_elements_index.query(search_area):
        if map_elements[index].intersects(house):
            return False

    # Check collision with other houses
    if list(houses_index.intersection(house.bounds)):
        return False

    return True

def assign_household_attributes(
    tract: pd.Series,
    income_ranges: List[Tuple[int, int]],
    size_index_dict: dict,
    workers_index_dict: dict
) -> Tuple[int, int, int, int]:
    """
    Assign synthetic household attributes using demographic weights from a census tract.

    Args:
        tract (pd.Series): A row from the census dataframe.
        income_ranges (List[Tuple[int, int]]): Income ranges for households.
        size_index_dict (dict): Mapping of household size to vehicle weight index range.
        workers_index_dict (dict): Mapping of worker count to vehicle weight index range.

    Returns:
        Tuple[int, int, int, int]: income, household_size, number_of_workers, vehicles
    """

    # Income assignment
    income_weights = np.array(tract["10k to 15k":"150k to 200k"]).astype(int)
    if income_weights.sum() == 0:
        raise ValueError("Income weights in tract are all zero.")

    income_range = random.choices(income_ranges, weights=income_weights)[0]
    income = random.randint(income_range[0] // 1000, income_range[1] // 1000) * 1000

    # Household size
    household_size = random.choices([1, 2, 3, 4, 5, 6, 7], weights=[1, 1, 1, 1, 0, 0, 0])[0]

    # Worker assignment
    worker_weights = [
        int(tract.get(col, 0)) if int(tract.get(col, 0)) != -666666666 else 0
        for col in [
            "1 Person(s) 0 Worker(s)", "1 Person(s) 1 Worker(s)",
            "2 Person(s) 0 Worker(s)", "2 Person(s) 1 Worker(s)", "2 Person(s) 2 Worker(s)",
            "3 Person(s) 0 Worker(s)", "3 Person(s) 1 Worker(s)", "3 Person(s) 2 Worker(s)", "3 Person(s) 3 Worker(s)",
            "4+ Person(s) 0 Worker(s)", "4+ Person(s) 1 Worker(s)", "4+ Person(s) 2 Worker(s)", "4+ Person(s) 3+ Worker(s)"
        ]
    ]

    if household_size == 1:
        num_workers = random.choices([0, 1], weights=worker_weights[:2])[0]
    elif household_size == 2:
        num_workers = random.choices([0, 1, 2], weights=worker_weights[2:5])[0]
    elif household_size == 3:
        num_workers = random.choices([0, 1, 2, 3], weights=worker_weights[5:9])[0]
    else:
        num_workers = random.choices([0, 1, 2, 3], weights=worker_weights[9:])[0]

    # Vehicle assignment
    vehicle_weights = [int(tract.get(col, 0)) if int(tract.get(col, 0)) != -666666666 else 0
                       for col in tract.index if "Vehicle(s)" in col]

    size_indexes = size_index_dict.get(min(household_size, 4), (0, 0))
    workers_indexes = workers_index_dict.get(num_workers, (0, 0))

    if workers_indexes[1] > workers_indexes[0]:
        combined_weights = (
            np.array(vehicle_weights[size_indexes[0]:size_indexes[1]]) +
            np.array(vehicle_weights[workers_indexes[0]:workers_indexes[1]])
        )
    else:
        combined_weights = np.array(vehicle_weights[size_indexes[0]:size_indexes[1]])

    vehicles = random.choices([0, 1, 2, 3, 4], weights=combined_weights)[0]

    return income, household_size, num_workers, vehicles



def generate_houses_from_housing_areas(
    housing_areas: List[Polygon],
    map_elements: List[Polygon],
    map_elements_index: STRtree,
    tract_index: STRtree,
    data: geopandas.GeoDataFrame,
    income_ranges: List[Tuple[int, int]],
    size_index_dict: dict,
    workers_index_dict: dict
) -> List[Tuple]:
    """
    Generate synthetic household records based on housing areas and census data.

    Args:
        housing_areas (List[Polygon]): List of buffered areas along residential roads.
        map_elements (List[Polygon]): List of roads and food store geometries.
        map_elements_index (STRtree): Spatial index for map elements.
        tract_index (STRtree): Spatial index of tract geometries.
        data (gpd.GeoDataFrame): GeoDataFrame of census tracts with household data.
        income_ranges (List[Tuple[int, int]]): Income range brackets.
        size_index_dict (dict): Index mapping from household size to vehicle bins.
        workers_index_dict (dict): Index mapping from worker count to vehicle bins.

    Returns:
        List[Tuple]: List of tuples ready for SQL insert into households table.
    """
    houses_index = rtree.index.Index()
    house_tuples: List[Tuple] = []
    total_count = 0

    for idx, housing_area in enumerate(housing_areas):
        logging.info(f"{round((idx + 1) / len(housing_areas) * 100)}%")

        candidate_houses = place_houses_in_area(housing_area)

        for house in candidate_houses:
            if not is_valid_house(house, map_elements_index, houses_index, map_elements):
                continue

            # Add to index for spatial collision checks
            houses_index.add(total_count, house.bounds)

            # Find matching tract for the house centroid
            tract_row = None
            for row_id in tract_index.query(house.centroid):
                if data.loc[row_id, "geometry"].contains(house.centroid):
                    tract_row = data.loc[row_id]
                    break

            if tract_row is None:
                continue  # Skip house if not inside any tract

            try:
                income, size, workers, vehicles = assign_household_attributes(
                    tract_row, income_ranges, size_index_dict, workers_index_dict
                )
            except Exception:
                continue  # skip malformed or invalid tract rows

            # Dummy travel times for now
            walking_time = biking_time = transit_time = driving_time = "NA"

            house_tuples.append((
                total_count,
                str(house),
                income,
                size,
                vehicles,
                workers,
                walking_time,
                biking_time,
                transit_time,
                driving_time
            ))

            total_count += 1

    return house_tuples


def get_nearest_store(
        house: Polygon, 
        store_tuples : List[Tuple[str, str, str]], 
        shapely_loader: Callable[[str], Polygon]
        )-> Optional[Polygon]:
    """
    Find the nearest store polygon to house. 

    Args:
        house (Polygon): The house polygon to check
        store_tuples: (List[Tuple[str, str, str]]): List of tuples (shop type, WKT polygon, name) for stores.
        shapely_loader (Any): Function to convert WKT string to Shapely geometry.
    
    Returns:
        Optional[Polygon]: The nearest store polygon, or None if no stores found.
    """
    nearest_store = None
    store_distance = float('inf')

    for store in store_tuples:
        store_poly = shapely_loader(store[1])
        dist = store_poly.distance(house)

        if dist <= store_distance:
            nearest_store = store_poly
            store_distance = DIST

    return nearest_store


def transform_polygon_coords(polygon: Polygon, source_crs : str, target_crs : str) -> Polygon:
    """Transform a polygon's coordinates from one CRS to another.

    Args:
        polygon (Polygon): The polygon to transform.
        source_crs (str): The source coordinate reference system (e.g., "EPSG:3857").
        target_crs (str): The target coordinate reference system (e.g., "EPSG:4326").

    Returns:
        Polygon: A new Polygon with its coordinates in the target CRS.
    """
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    coords = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
    return Polygon(coords)

def get_tract_for_house(
    house: Polygon,
    tract_index: STRtree,
    data: pd.DataFrame
) -> Optional[pd.Series]:
    """
    Find the census tract containing the house centroid.

    Args:
        house (Polygon): The house polygon.
        tract_index (STRtree): Spatial index of tract geometries.
        data (pd.DataFrame): Census data including geometries.

    Returns:
        Optional[pd.Series]: The tract row containing the house, or None.
    """
    for row_id in tract_index.query(house.centroid):
        if data.loc[row_id, "geometry"].contains(house.centroid):
            return data.loc[row_id]
    return None

def process_housing_areas(
    housing_areas: List[Polygon],
    map_elements_index: STRtree,
    map_elements: List[Polygon],
    houses_index: RTreeIndex,
    tract_index: STRtree,
    data: pd.DataFrame,
    income_ranges: List[Tuple[int, int]],
    size_index_dict: dict,
    workers_index_dict: dict,
    vehicle_weights: List[int],
    worker_weights: List[int],
    store_tuples: List[Tuple[str, str, str]],
    shapely_loader: Any
) -> List[Tuple]:
    """
    Process a list of housing area polygons, generate synthetic houses along their borders,
    assign attributes using tract data, and return ready-to-insert household tuples.

    Args:
        housing_areas (List[Polygon]): Polygons representing areas around residential roads.
        map_elements_index (STRtree): STRtree index for nearby road/store geometries.
        map_elements (List[Polygon]): Buffered geometries (roads, stores) for intersection checking.
        houses_index (RTreeIndex): R-tree index to check house overlap.
        tract_index (STRtree): STRtree index of tract geometries.
        data (pd.DataFrame): Census tract data with attributes.
        income_ranges (List[Tuple[int, int]]): Bracketed income ranges for sampling.
        size_index_dict (dict): Dict for mapping household sizes to vehicle weight indices.
        workers_index_dict (dict): Dict for mapping worker counts to vehicle weight indices.
        vehicle_weights (List[int]): Vehicle weights from census.
        worker_weights (List[int]): Worker weights from census.
        store_tuples (List[Tuple[str, str, str]]): Store (shop type, WKT, name).
        shapely_loader (Any): Function to load WKT strings into Polygons.

    Returns:
        List[Tuple]: Tuples of household data ready for DB insertion.
    """
    house_tuples: List[Tuple] = []
    total_count = 0

    for i, housing_area in enumerate(housing_areas):
        logging.info(f"{round((i + 1) / len(housing_areas) * 100)}%")

        exterior_coords = list(housing_area.exterior.coords)
        edges = [LineString([exterior_coords[i], exterior_coords[i + 1]])
                 for i in range(len(exterior_coords) - 1)]

        for edge in edges:
            direction = (edge.coords[1][0] - edge.coords[0][0], edge.coords[1][1] - edge.coords[0][1])
            magnitude = math.hypot(*direction)
            norm_vector = (direction[0] / magnitude, direction[1] / magnitude) if magnitude else (0, 0)

            for j in range(int(magnitude // 30) + 1):
                location = Point(
                    edge.coords[0][0] + norm_vector[0] * j * 30,
                    edge.coords[0][1] + norm_vector[1] * j * 30
                )

                house = create_house_polygon(location)

                if not is_valid_house(house, map_elements_index, houses_index, map_elements):
                    continue

                houses_index.add(total_count, house.bounds)

                tract = get_tract_for_house(house, tract_index, data)
                if tract is None:
                    continue

                try:
                    income, size, workers, vehicles = assign_household_attributes(
                        tract, income_ranges, size_index_dict, workers_index_dict,
                        vehicle_weights, worker_weights
                    )
                except Exception:
                    continue

                nearest_store = get_nearest_store(house, store_tuples, shapely_loader)
                if nearest_store is None:
                    continue

                house_4326 = transform_polygon_coords(house, "EPSG:3857", "EPSG:4326")
                store_4326 = transform_polygon_coords(nearest_store, "EPSG:3857", "EPSG:4326")
                origin = (float(house_4326.centroid.y), float(house_4326.centroid.x))
                destination = (float(store_4326.centroid.y), float(store_4326.centroid.x))

                # Placeholder travel times (to be replaced with real data if available)
                walking_time = biking_time = transit_time = driving_time = 0

                house_tuples.append((
                    total_count,
                    str(house),
                    income,
                    size,
                    vehicles,
                    workers,
                    walking_time,
                    biking_time,
                    transit_time,
                    driving_time
                ))

                total_count += 1

    return house_tuples


def insert_households(cursor, house_tuples: List[Tuple], household_query: str) -> None:
    """
    Bulk insert generated data into the database.

    Args:
        cursor: database cursor used for executing SQL commands
        house_tuples (List[Tuple]): list of household tuples to be inserted
        household_query (str): The SQL query template for inserting households

    Raises:
        Exception: If insertion fails, the exception is re-raised after logging.infoing the error.
    """
    try:
        extras.execute_values(cursor, household_query, house_tuples)
    except Exception as e:
        logging.info(f"Insertion error: {e}")
        raise


def connect_to_db(
    host: str,
    database: str,
    user: str,
    password: str,
    port: str
) -> Tuple[Connection, Cursor]:
    """
    Establish connection to the PostgreSQL database.

    Args:
        host (str): Database host.
        database (str): Database name.
        user (str): Username.
        password (str): Password.
        port (str): Port number.

    Returns:
        Tuple[Connection, Cursor]: The DB connection and cursor.
    """
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port
    )
    return conn, conn.cursor()

def main() -> None:
    """
    Main function to execute the full data processing and insertion pipeline.
    """
    logging.info("Fetching household census data...")
    county_data = fetch_county_data(households_key_list, YEAR, STATE_CODE, COUNTY_CODE, APIKEY)

    logging.info("Loading and merging tract shapefile with census data...")
    tract_data = load_and_merge_geodata(YEAR, STATE_CODE, COUNTY_CODE, county_data)

    logging.info("Initializing database and creating tables...")
    connection, cursor = initialize_database_tables(HOST, NAME, USER, PASS, PORT)
    household_query = create_households_table(cursor)

    logging.info("Processing road network...")
    map_elements, housing_areas, road_tuples = process_road_network(CENTER_POINT, DIST)

    logging.info("Processing food stores...")
    store_index = process_food_stores(CENTER_POINT, DIST, map_elements, cursor)

    logging.info("Generating household records...")
    tract_index = STRtree(tract_data["geometry"])
    houses_index = RTreeIndex()
    store_tuples = list(cursor.execute("SELECT shop, geometry, name FROM food_stores;"))  # fallback in case needed

    house_tuples = process_housing_areas(
        housing_areas,
        store_index,
        map_elements,
        houses_index,
        tract_index,
        tract_data,
        income_ranges,
        size_index_dict,
        workers_index_dict,
        [],  # vehicle_weights are computed inside assign function
        [],  # worker_weights too
        store_tuples,
        shapely.wkt.loads
    )

    logging.info("Inserting households into the database...")
    insert_households(cursor, house_tuples, household_query)

    logging.info("Finalizing and closing connection...")
    connection.commit()
    cursor.close()
    connection.close()
    logging.info("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(F"Fatal error in pipeline: {e}", exc_info=True)
        sys.exit(1)
