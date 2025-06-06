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
from household_constants import(
    households_variables_dict,
    households_key_list,
    FIPSCODE,
    YEAR,
    income_ranges,
    size_index_dict,
    workers_index_dict
)

place_name = "Franklin County, Ohio, USA"    

county_code = FIPSCODE[2:]                  
state_code = FIPSCODE[:2]                  

center_point = (39.938806, -82.972361)      
dist = 1000

PASS = os.getenv("DB_PASS")
APIKEY = os.getenv("APIKEY")
USER = os.getenv("DB_USER")
NAME = os.getenv("DB_NAME")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")

county_data = pd.DataFrame()

for count in range(int(len(households_key_list)/50)+1):
    variables = ""
    if ((count+1)*50) > len(households_key_list):
        variables = ",".join(households_key_list[(50*count):])
    elif count == 0:
        if (int(len(households_key_list)/50)+1) == 1:
            variables = ",".join(households_key_list[:])
        else:
            variables = ",".join(households_key_list[:(50*(count+1)-1)])
    else:
        variables = ",".join(households_key_list[(50*count):(50*(count+1)-1)])

    url = f"https://api.census.gov/data/{YEAR}/acs/acs5?get=NAME,
    {variables}&for=tract:*&in=state:{state_code}&in=county:{county_code}&key={APIKEY}"

    response = requests.request("GET", url)
    if len(county_data != 0):
        county_data = pd.merge(
            pd.DataFrame(response.json()[1:], 
            columns=response.json()[0]), 
            county_data, 
            on='NAME', 
            how='inner')
    else:
         # If this is the first batch, initialize the dataset
        county_data = pd.DataFrame(response.json()[1:], columns=response.json()[0])

tract_url = f"https://www2.census.gov/geo/tiger/TIGER{YEAR}/TRACT/tl_{YEAR}_{state_code}_tract.zip"
response = requests.request("GET", tract_url)
with ZipFile(BytesIO(response.content)) as zip_ref:
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_ref.extractall(tmpdirname)
        
        # Loop through the extracted files to locate the .shp or .geojson
        for root, dirs, files in os.walk(tmpdirname):
            for file in files:
                if file.endswith(".shp") or file.endswith(".geojson"):
                    file_path = os.path.join(root, file)
                    # Load shapefile/geojson into a GeoDataFrame and convert to EPSG:3857 projection
                    geodata = (geopandas.read_file(file_path)).to_crs("epsg:3857")


county_geodata = geodata[geodata['COUNTYFP'] == county_code]
county_geodata = county_geodata.rename(columns={"TRACTCE":"tract_y"})
county_geodata["tract_y"] = county_geodata["tract_y"].astype(int)
county_data["tract_y"] = county_data["tract_y"].astype(int)

data = pd.merge(county_geodata, county_data, on = "tract_y", how="inner")
data.rename(columns=households_variables_dict, inplace = True)

# Convert all geometries to EPSG:3857 projection (used for consistent spatial analysis)
data = data.to_crs("epsg:3857")
tract_index = STRtree(data["geometry"])


# Connect to the PostgreSQL database using credentials from environment variables
connection = psycopg2.connect(
    host=HOST,
    database=NAME,
    user=USER,
    password=PASS,
    port=PORT
)
cursor = connection.cursor()

cursor.execute('DROP TABLE IF EXISTS roads;')
cursor.execute('DROP TABLE IF EXISTS food_stores;')

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

place_name = "Franklin County, Ohio, USA"
map_elements = list()  
housing_areas = list() 

G = ox.graph_from_point(center_point,dist=dist, network_type='all',retain_all=True)
# Convert the OSM road graph into GeoDataFrames: one for nodes, one for edges
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

# Convert the coordinate reference system to Web Mercator (meters)
gdf_edges = gdf_edges.to_crs("epsg:3857")

if "service" in gdf_edges.columns:
    gdf_edges = gdf_edges[["name","highway","length","geometry","service"]]
else:
    gdf_edges = gdf_edges[["name","highway","length","geometry"]]

# Loop through each road segment to categorize and apply appropriate buffer sizes
for index,row in gdf_edges.iterrows():
    if (row["highway"] == "residential") or (row["highway"] == "living_street"):
        housing_areas.append(row["geometry"].buffer(30))    # Wider buffer to simulate houses
        map_elements.append((row["geometry"]).buffer(3))    # Narrow buffer to store the road outline
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

features = ox.features.features_from_point(
    center_point,
    dist=dist*3,
    tags = {"shop":[
        "convenience",
    'supermarket',
    "butcher",
    "wholesale",
    "farm",
    "greengrocer",
    "health_food",
    "grocery"]})

features = features.to_crs("epsg:3857")
features = features[["shop","geometry","name"]]

store_tuples = list()
food_stores_query = "INSERT INTO food_stores (shop,geometry,name) VALUES %s"

# Loop through each store and create a polygon representation for its location
for index, row in features.iterrows():
    # Use the centroid if geometry is not a simple point
    point = row["geometry"].centroid if not isinstance(row["geometry"], Point) else row["geometry"]

    if row["shop"] in ["supermarket", "grocery", "greengrocer"]:
        # Generate a hexagonal polygon for large stores like supermarkets
        polygon = Polygon([
            (point.x + 50 * math.cos(math.radians(angle)), point.y + 50 * math.sin(math.radians(angle)))
            for angle in range(0, 360, 60)
        ])
    else:
        # Generate a triangular-style shape for small shops
        polygon = Polygon([
            (point.x, point.y + 20),
            (point.x + 25, point.y - 30),
            (point.x - 25, point.y - 30)
        ])

    # Buffer the polygon and add it to the spatial map elements
    map_elements.append(polygon.buffer(20))

    # Add the store's details to the list for database insertion
    store_tuples.append((str(row["shop"]), str(polygon), str(row["name"])))

# Build a spatial index for fast lookup of all mapped elements
map_elements_index = STRtree(map_elements)

# Insert all food store records into the database in bulk
extras.execute_values(cursor, food_stores_query, store_tuples)

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
driving_time) VALUES %s"
"""

# Commit changes and close DB connection
connection.commit()
cursor.close()
connection.close()

# This whole thing creates houses and assigns attributes to them, and then stores the households in a SQL DB
houses = list()
houses_index = rtree.index.Index()
total_count = 0
total_google_pulls = 0
house_tuples = list()
housing_areas_count = 0
#Iterate through each road and place houses next to the road (housing area means aread around a residential road)
for housing_area in housing_areas:
    housing_areas_count+=1
    print(str(round(housing_areas_count/len(housing_areas)*100)) + "%")
    #print(total_google_pulls)
    count = 0
    # Get the exterior coordinates of the polygon
    exterior_coords = list(housing_area.exterior.coords)
    # Create LineStrings for each edge
    edges = [LineString([exterior_coords[i], exterior_coords[i+1]]) 
            for i in range(len(exterior_coords) - 1)]
        
    # Basically, draw a polygon around each road and then place houses on the edges of that rectangle
    for edge in edges:
        # Calculate the vector representation of each edge of the polygon
        length = edge.length
        coord1 = edge.coords[0]
        coord2 = edge.coords[1]
        vector_direction = (coord2[0] - coord1[0], coord2[1] - coord1[1])
        temp = (vector_direction[0])*(vector_direction[0]) + (vector_direction[1])*(vector_direction[1])
        vector_magnitude = math.sqrt(temp)
        normalized_vector = (0,0)
        if vector_magnitude != 0:
            normalized_vector = (vector_direction[0]/vector_magnitude,vector_direction[1]/vector_magnitude)
        
        #place houses on the vector, each 20 meters away from each other until the vector ends
        for i in range(int(vector_magnitude/20)+1):
            location = Point(coord1[0]+normalized_vector[0]*i*30,coord1[1]+normalized_vector[1]*i*30)

            house = Polygon(((location.x+10, location.y+10),
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
                            (location.x+10, location.y-5)))
            
            
            valid = True

            # Get all nearby elements in map to check that house is not on top of anything
            intersecting_map_elements_indexes = map_elements_index.query(Polygon(((location.x+10, location.y+20),
                                                                        (location.x-10, location.y+20),
                                                                        (location.x-10, location.y-5),
                                                                        (location.x+10, location.y-5))))
            
            # Check if house is on an element
            if len(intersecting_map_elements_indexes) != 0:
                for index in intersecting_map_elements_indexes:
                    if map_elements[index].intersects(house):
                        valid=False
                        continue
            if not valid:
                continue

            # Check if house is touching another house
            intersecting_houses_indexes = houses_index.intersection(house.bounds)
            if len(list(intersecting_houses_indexes)) != 0:
                continue

            houses.append(house)
            # This is a special index that makes it faster to query houses when we check if houses are on top of each other
            houses_index.add(total_count,house.bounds)
            
            # Get the tract that the house is in
            tract_row_numbers = tract_index.query(house.centroid)
            tract_row = 0
            for row_number in tract_row_numbers:
                if (data.loc[row_number,"geometry"]).contains(house.centroid):
                    tract_row = row_number
                    break
            tract = data.loc[tract_row]

            #Iterate through each tract and create households
            income_weights = np.array(tract["10k to 15k":"150k to 200k"]).astype(int)
            if sum(income_weights)==0:
                continue

            # Make a list of incomes to distribute to households
            total_households = int(tract["total households in tract"])
            distributed_incomes = []
            for i in range(14):
                uniform_list = []
                if i != 14:
                    uniform_list = np.random.uniform(income_ranges[i][0],income_ranges[i][1],income_weights[i])
                else:
                    uniform_list = np.random.uniform(200000,200000,income_weights[i])
                distributed_incomes.extend(uniform_list.astype(int))

            vehicle_weights = [
                                int(tract["1 Person(s) 0 Vehicle(s)"]),
                                int(tract["1 Person(s) 1 Vehicle(s)"]),
                                int(tract["1 Person(s) 2 Vehicle(s)"]),
                                int(tract["1 Person(s) 3 Vehicle(s)"]),
                                int(tract["1 Person(s) 4+ Vehicle(s)"]),
                                int(tract["2 Person(s) 0 Vehicle(s)"]),
                                int(tract["2 Person(s) 1 Vehicle(s)"]),
                                int(tract["2 Person(s) 2 Vehicle(s)"]),
                                int(tract["2 Person(s) 3 Vehicle(s)"]),
                                int(tract["2 Person(s) 4+ Vehicle(s)"]),
                                int(tract["3 Person(s) 0 Vehicle(s)"]),
                                int(tract["3 Person(s) 1 Vehicle(s)"]),
                                int(tract["3 Person(s) 2 Vehicle(s)"]),
                                int(tract["3 Person(s) 3 Vehicle(s)"]),
                                int(tract["3 Person(s) 4+ Vehicle(s)"]),
                                int(tract["4+ Person(s) 0 Vehicle(s)"]),
                                int(tract["4+ Person(s) 1 Vehicle(s)"]),
                                int(tract["4+ Person(s) 2 Vehicle(s)"]),
                                int(tract["4+ Person(s) 3 Vehicle(s)"]),
                                int(tract["4+ Person(s) 4+ Vehicle(s)"]),
                                int(tract["0 Worker(s) 0 Vehicle(s)"]),
                                int(tract["0 Worker(s) 1 Vehicle(s)"]),
                                int(tract["0 Worker(s) 2 Vehicle(s)"]),
                                int(tract["0 Worker(s) 3 Vehicle(s)"]),
                                int(tract["0 Worker(s) 4+ Vehicle(s)"]),
                                int(tract["1 Worker(s) 0 Vehicle(s)"]),
                                int(tract["1 Worker(s) 1 Vehicle(s)"]),
                                int(tract["1 Worker(s) 2 Vehicle(s)"]),
                                int(tract["1 Worker(s) 3 Vehicle(s)"]),
                                int(tract["1 Worker(s) 4+ Vehicle(s)"]),
                                int(tract["2 Worker(s) 0 Vehicle(s)"]),
                                int(tract["2 Worker(s) 1 Vehicle(s)"]),
                                int(tract["2 Worker(s) 2 Vehicle(s)"]),
                                int(tract["2 Worker(s) 3 Vehicle(s)"]),
                                int(tract["2 Worker(s) 4+ Vehicle(s)"]),
                                int(tract["3+ Worker(s) 0 Vehicle(s)"]),
                                int(tract["3+ Worker(s) 1 Vehicle(s)"]),
                                int(tract["3+ Worker(s) 2 Vehicle(s)"]),
                                int(tract["3+ Worker(s) 3 Vehicle(s)"]),
                                int(tract["3+ Worker(s) 4+ Vehicle(s)"])
                            ]
            vehicle_weights = [0 if item == -666666666 else item for item in vehicle_weights]

            worker_weights = [
                                int(tract["1 Person(s) 0 Worker(s)"]),
                                int(tract["1 Person(s) 1 Worker(s)"]),
                                int(tract["2 Person(s) 0 Worker(s)"]),
                                int(tract["2 Person(s) 1 Worker(s)"]),
                                int(tract["2 Person(s) 2 Worker(s)"]),
                                int(tract["3 Person(s) 0 Worker(s)"]),
                                int(tract["3 Person(s) 1 Worker(s)"]),
                                int(tract["3 Person(s) 2 Worker(s)"]),
                                int(tract["3 Person(s) 3 Worker(s)"]),
                                int(tract["4+ Person(s) 0 Worker(s)"]),
                                int(tract["4+ Person(s) 1 Worker(s)"]),
                                int(tract["4+ Person(s) 2 Worker(s)"]),
                                int(tract["4+ Person(s) 3+ Worker(s)"]),
                            ]
            worker_weights = [0 if item == -666666666 else item for item in worker_weights]

            household_size_weights = [
                                int(tract["Median Income for 1 Person(s)"]),
                                int(tract["Median Income for 2 Person(s)"]),
                                int(tract["Median Income for 3 Person(s)"]),
                                int(tract["Median Income for 4 Person(s)"]),
                                int(tract["Median Income for 5 Person(s)"]),
                                int(tract["Median Income for 6 Person(s)"]),
                                int(tract["Median Income for 7+ Person(s)"])
                            ]
            household_size_weights = [0 if item == -666666666 else item for item in household_size_weights]
            income_range = random.choices(income_ranges,weights = income_weights)
            income = random.randint(int(income_range[0][0]/1000),int(income_range[0][1]/1000))*1000

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

def main():
    """
    Main execution function 
    """
     # TODO: After creating functions call them in here

if __name__ == "__main__":
    main()
