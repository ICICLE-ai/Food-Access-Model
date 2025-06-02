import osmnx as ox
import psycopg2
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
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
from io import BytesIO
from typing import List, Tuple, Optional, Any
from rtree.index import Index
from household_constants import(
    households_variables_dict,
    households_key_list,
    FIBSCODE,
    YEAR,
    income_ranges,
    size_index_dict,
    workers_index_dict
)



place_name = "Franklin County, Ohio, USA"

county_code = FIBSCODE[2:]
state_code = FIBSCODE[:2]

center_point = (39.938806, -82.972361)
dist = 1000

PASS = os.getenv("DB_PASS")
APIKEY = os.getenv("APIKEY")
USER = os.getenv("DB_USER")
NAME = os.getenv("DB_NAME")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")

#Read csvs into pandas dataframes
#For loop runs a census API pull for each loop iteration
#this is neccessary because we can only pull 50 variables at a time and we have >50
county_data = pd.DataFrame()
for count in range(int(len(households_key_list)/50)+1):
    variables = ""
    #put variables from above into census readable lists
    if ((count+1)*50) > len(households_key_list):
        variables = ",".join(households_key_list[(50*count):])
    elif count == 0:
        if (int(len(households_key_list)/50)+1) == 1:
            variables = ",".join(households_key_list[:])
        else:
            variables = ",".join(households_key_list[:(50*(count+1)-1)])
    else:
        variables = ",".join(households_key_list[(50*count):(50*(count+1)-1)])
    #Pull data and add to dataframe
    url = f"https://api.census.gov/data/{YEAR}/acs/acs5?get=NAME,{variables}&for=tract:*&in=state:{state_code}&in=county:{county_code}&key={APIKEY}"
    response = requests.request("GET", url)
    if len(county_data != 0):
        county_data = pd.merge(pd.DataFrame(response.json()[1:], columns=response.json()[0]), county_data, on='NAME', how='inner')
    else:
        county_data = pd.DataFrame(response.json()[1:], columns=response.json()[0])


# Load in geographical tract data
tract_url = f"https://www2.census.gov/geo/tiger/TIGER{YEAR}/TRACT/tl_{YEAR}_{state_code}_tract.zip"
response = requests.request("GET", tract_url)
# Use BytesIO to handle the zip file in memory
with ZipFile(BytesIO(response.content)) as zip_ref:
    # Create a temporary directory to extract the zip file
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_ref.extractall(tmpdirname)
        
        # Find the shapefile or GeoJSON file in the extracted contents
        for root, dirs, files in os.walk(tmpdirname):
            for file in files:
                if file.endswith(".shp") or file.endswith(".geojson"):
                    file_path = os.path.join(root, file)
                    # Load the file into a GeoDataFrame
                    geodata = (geopandas.read_file(file_path)).to_crs("epsg:3857")


#Merge geographical dataframe (containing shapely ploygons) with census data
county_geodata = geodata[geodata['COUNTYFP'] == county_code]
county_geodata = county_geodata.rename(columns={"TRACTCE":"tract_y"})
county_geodata["tract_y"] = county_geodata["tract_y"].astype(int)
county_data["tract_y"] = county_data["tract_y"].astype(int)
data = pd.merge(county_geodata, county_data, on = "tract_y", how="inner")
data.rename(columns=households_variables_dict, inplace = True)
data = data.to_crs("epsg:3857")
tract_index = STRtree(data["geometry"])

# Connect to the PostgreSQL database
connection = psycopg2.connect(
    host=HOST,
    database=NAME,
    user=USER,
    password=PASS,
    port=PORT
)
cursor = connection.cursor()

# Execute the drop table command for roads
cursor.execute('DROP TABLE IF EXISTS roads;')

# Execute the drop table command for stores
cursor.execute('DROP TABLE IF EXISTS food_stores;')

# SQL query to create the 'roads' table
create_roads_query = '''
CREATE TABLE roads (
    name TEXT,
    highway VARCHAR(30),
    length NUMERIC,
    geometry TEXT,
    service VARCHAR(30)
);
'''

# SQL query to create the 'roads' table
create_food_stores_query = '''
CREATE TABLE food_stores (
    shop VARCHAR(15),
    geometry TEXT,
    name VARCHAR(50)
);
'''

# Execute the create table command
cursor.execute(create_roads_query)

# Execute the create table command
cursor.execute(create_food_stores_query)

place_name = "Franklin County, Ohio, USA"
map_elements = list()
housing_areas = list()

#Get road network from open street maps
G = ox.graph_from_point(center_point,dist=dist, network_type='all',retain_all=True)
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

#convert to epsg:3857
gdf_edges = gdf_edges.to_crs("epsg:3857")
if "service" in gdf_edges.columns:
    gdf_edges = gdf_edges[["name","highway","length","geometry","service"]]
else:
    gdf_edges = gdf_edges[["name","highway","length","geometry"]]

# Insert data into the table using a SQL query
for index,row in gdf_edges.iterrows():
    if (row["highway"] == "residential") or (row["highway"] == "living_street"):
        housing_areas.append(row["geometry"].buffer(30))
        map_elements.append((row["geometry"]).buffer(3))
    elif ("service" in gdf_edges.columns) and ((row["service"])=="alley"):
        housing_areas.append(row["geometry"].buffer(30))
        map_elements.append((row["geometry"]).buffer(3))
    elif (row["highway"] == "motorway"):
        map_elements.append((row["geometry"]).buffer(100))
    elif (row["highway"] == "trunk"):
        map_elements.append((row["geometry"]).buffer(30))
    elif (row["highway"] == "primary"):
        map_elements.append((row["geometry"]).buffer(10))
    elif (row["highway"] == "secondary"):
        map_elements.append((row["geometry"]).buffer(10))
    elif isinstance((row["geometry"]), LineString):
        map_elements.append((row["geometry"]))

gdf_edges["length"] = gdf_edges["length"].astype(int)
gdf_edges["geometry"] = gdf_edges["geometry"].astype(str)
roads_query = "INSERT INTO roads (name,highway,length,geometry,service) VALUES %s"
data_tuples = list(gdf_edges.itertuples(index=False, name=None))
#extras.execute_values(cursor, roads_query, data_tuples)

#Get food stores
features = ox.features.features_from_point(center_point,dist=dist*3,tags = {"shop":["convenience",'supermarket',"butcher","wholesale","farm",'greengrocer',"health_food",'grocery']})
features = features.to_crs("epsg:3857")
features = features[["shop","geometry","name"]]

#Insert food stores into postgres database
store_tuples = list()
food_stores_query = "INSERT INTO food_stores (shop,geometry,name) VALUES %s"
for index,row in features.iterrows():
    point = None
    if not isinstance(row["geometry"],Point):
        point = row["geometry"].centroid
    else:
        point = row["geometry"]

    if (row["shop"] == "supermarket") or (row["shop"]=="grocery") or (row["shop"]=="greengrocer"):
        polygon = Polygon([(point.x + 50 * math.cos(math.radians(angle)), point.y + 50 * math.sin(math.radians(angle))) for angle in range(0, 360, 60)])
    else:
        polygon = Polygon([
            (point.x, point.y + 20),           # Top vertex
            (point.x + 25, point.y - 30),      # Bottom right vertex
            (point.x - 25, point.y - 30)       # Bottom left vertex
        ])
    map_elements.append(polygon.buffer(20))
    store_tuples.append((str(row["shop"]),str(polygon),str(row["name"])))

map_elements_index = STRtree(map_elements)
extras.execute_values(cursor, food_stores_query, store_tuples)

# SQL query to create the 'households' table
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

# Execute the drop table command
cursor.execute('DROP TABLE IF EXISTS households;')

# Execute the create table command
cursor.execute(create_households_query)

household_query = "INSERT INTO households (id,polygon,income,household_size,vehicles,number_of_workers,walking_time,biking_time,transit_time,driving_time) VALUES %s"

connection.commit()
cursor.close()
connection.close()


def create_house_polygon(location : Point) -> Polygon:
    """
    Creates a polygon that represents a house at the given center

    Args:
        location (Point): The center point where the house polygon will be created
    Returns: 
        Polygon: a shapely polygon representing the house
    """
    return Polygon([
        (location.x+10, location.y+10),
        (location.x, location.y+20),
        (location.x-10, location.y+10),
        (location.x+10, location.y+10),
        (location.x-10, location.y+10),
        (location.x-10, location.y-5),
        (location.x-3, location.y-5),
        (location.x-3, location.y+3),
        (location.x+3, location.y+3),
        (location.x+3, location.y-5),
        (location.x-3, location.y-5),
        (location.x+10, location.y-5)
    ])

def is_house_location_valid( house: Polygon, map_elements_index: STRtree, map_elements: List[Polygon], houses_index: Index) -> bool:
    """
    Checks if a house is not on top/does not intersect another map element or another house

    Args: 
        house (Polygon): The house polygon to check
        map_elements_index (STRtree): spatial index for map elements
        map_elements List[Polygon]: list of map element polygons
        house_index (Index): R-tree index of already placed house
    Returns:
        bool: true if house location is valid else false.
    """
    for idx in map_elements_index.query(house):
        if map_elements[idx].intersects(house):
            return False
    if any(houses_index.intersection(house.bounds)):
        return False
    return True

def get_tract_for_house(house: Polygon, tract_index: STRtree,data: pd.DataFrame) -> Optional[pd.Series]:
    """
    Gets the census tract that contains house centroid

    Args: 
        house (polygon): the house polygon to check
        tract_index (STRtree): spatial index for census tracts
        data (pd.DataFrame): dataframe containing tract geometries and data
    
    Returns:
        Optional[pd.Series]: The row of the DataFrame coressponding to the containing tract
    """
    for row_number in tract_index.query(house.centroid):
        if data.loc[row_number, "geometry"].contains(house.centroid):
            return data.loc[row_number]
    return None

def assign_household_attributes(tract: pd.Series, income_ranges: List[Tuple[int, int]], size_index_dict: dict, workers_index_dict: dict, vehicle_weights: List[int], worker_weights: List[int]) -> Tuple[int, int, int, int]:
    """
    Assign income, household size, number of workers, and number of vechicles

    Args:
        tract (pd.Series): census tract row with demographic data.
        income_ranges (List[Tuple[int, int]]): list of possible income ranges.
        size_index_dict (dict): dict mapping household size to weight index ranges.
        workers_index_dict (dict): dict mapping number of workers to weight index ranges.
        vehicle_weights (List[int]): weights for assigning number of vehicles.
        worker_weights (List[int]): weights for assigning number of workers.

    Returns:
        Tuple[int, int, int, int]: (income, household_size, num_workers, vehicles)
    """
    income_weights = np.array(tract["10k to 15k":"150k to 200k"]).astype(int)
    if sum(income_weights) == 0:
        income = 0
    else:
        income_range = random.choices(income_ranges, weights=income_weights)[0]
        income = random.randint(int(income_range[0] / 1000), int(income_range[1] / 1000)) * 1000

    household_size = random.choices([1, 2, 3, 4, 5, 6, 7], weights=[1, 1, 1, 1, 0, 0, 0])[0]
    if household_size == 1:
        num_workers = random.choices([0, 1], weights=worker_weights[:2], k=1)[0]
    elif household_size == 2:
        num_workers = random.choices([0, 1, 2], weights=worker_weights[2:5], k=1)[0]
    elif household_size == 3:
        num_workers = random.choices([0, 1, 2, 3], weights=worker_weights[5:9], k=1)[0]
    else:
        num_workers = random.choices([0, 1, 2, 3], weights=worker_weights[9:], k=1)[0]

    size_indexes = size_index_dict[min(household_size, 4)]
    workers_indexes = workers_index_dict.get(num_workers, (0, 0))
    if workers_indexes[1] > workers_indexes[0]:
        vehicle_combined_weights = (
            np.array(vehicle_weights[size_indexes[0]:size_indexes[1]])
            + np.array(vehicle_weights[workers_indexes[0]:workers_indexes[1]])
        )
    else:
        vehicle_combined_weights = np.array(vehicle_weights[size_indexes[0]:size_indexes[1]])
    vehicles = random.choices([0, 1, 2, 3, 4], weights=vehicle_combined_weights)[0]
    return income, household_size, num_workers, vehicles

def get_nearest_store(house: Polygon, store_tuples : List[Tuple[str, str, str]], shapely_loader: Any) -> Optional[Polygon]:
    """
    Find the nearest store polygon to house. 

    Args:
        house (polygon): The house polygon to check
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
            store_distance = dist
    return nearest_store

def transform_polygon_coords(polygon: Polygon, source_crs : str, target_crs : str,) -> Polygon:
    """
    Function to transform polygon coordinates to another CRS
    """
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    coords = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
    return Polygon(coords)

def process_housing_areas(
        housing_areas: List[Polygon],
    map_elements_index: STRtree,
    map_elements: List[Polygon],
    houses_index: Index,
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
    Process housing areas and generate household tuples. Make sure the Args/data are defined somewhere!

    Args:
        housing_areas (List[Polygon]): housing area polygons
        map_elements_index (STRtree): Map elements spatial index
        map_elements (List[Polygon]): Map element polygons
        houses_index (Index): R-tree for houses
        tract_index (STRtree): tract index
        data (pd.DataFrame): Tract data.
        income_ranges (List[Tuple[int, int]]): income ranges
        size_index_dict (dict): Size index dict
        workers_index_dict (dict): Workers index dict
        vehicle_weights (List[int]): Vehicle weights
        worker_weights (List[int]): Worker weights.
        store_tuples (List[Tuple[str, str, str]]): Store tuples
        shapely_loader (Any): function to load WKT

    Returns:
        List[Tuple]: List of household tuples for insertion.
    """
    houses = []
    house_tuples = []
    total_count = 0
    housing_areas_count = 0

    for housing_area in housing_areas:
        housing_areas_count += 1
        print(f"{round(housing_areas_count/len(housing_areas)*100)}%")

        exterior_coords = list(housing_area.exterior.coords)
        edges = [LineString([exterior_coords[i], exterior_coords[i+1]])
                 for i in range(len(exterior_coords) - 1)]

        for edge in edges:
            vector_direction = (edge.coords[1][0] - edge.coords[0][0], edge.coords[1][1] - edge.coords[0][1])
            vector_magnitude = math.sqrt(vector_direction[0] ** 2 + vector_direction[1] ** 2)
            normalized_vector = (
                vector_direction[0] / vector_magnitude,
                vector_direction[1] / vector_magnitude
            ) if vector_magnitude else (0, 0)

            for i in range(int(vector_magnitude / 20) + 1):
                location = Point(
                    edge.coords[0][0] + normalized_vector[0] * i * 30,
                    edge.coords[0][1] + normalized_vector[1] * i * 30
                )
                house = create_house_polygon(location)
                if not is_house_location_valid(house, map_elements_index, map_elements, houses_index):
                    continue
                houses.append(house)
                houses_index.add(total_count, house.bounds)

                tract = get_tract_for_house(house, tract_index, data)
                if tract is None:
                    continue

                income, household_size, num_workers, vehicles = assign_household_attributes(
                    tract, income_ranges, size_index_dict, workers_index_dict,
                    vehicle_weights, worker_weights
                )

                nearest_store = get_nearest_store(house, store_tuples, shapely_loader)
                if nearest_store is None:
                    continue

                house_4326 = transform_polygon_coords(house, "EPSG:3857", "EPSG:4326")
                store_4326 = transform_polygon_coords(nearest_store, "EPSG:3857", "EPSG:4326")
                origin = (float(house_4326.centroid.y), float(house_4326.centroid.x))
                destination = (float(store_4326.centroid.y), float(store_4326.centroid.x))

                # Placeholders for travel times 
                walking_time = biking_time = transit_time = driving_time = 0

                house_tuples.append((
                    total_count, str(house), income, household_size, vehicles, num_workers,
                    walking_time, biking_time, transit_time, driving_time
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
        Exception: If insertion fails, the exception is re-raised after printing the error.
    """
    try:
        extras.execute_values(cursor, household_query, house_tuples)
    except Exception as e:
        print(f"Insertion error: {e}")
        raise



# Connect to the PostgreSQL database
connection = psycopg2.connect(
    host=HOST,
    database=NAME,
    user=USER,
    # password=PASS,
    port=PORT
)
cursor = connection.cursor()

extras.execute_values(cursor, household_query, house_tuples)

# Close the cursor and connection
connection.commit()
cursor.close()
connection.close()

