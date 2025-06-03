
"""
get_data.py

This file collects, processes, and stores household, road, and food store data
for Franklin County, Ohio using data from OSM and the US Census API.

Author: Abanish and Shrivas 
Date: 31st May, 2025
"""

# importing required libraries, lists and dictionary.
import osmnx as ox                    # For retrieving and analyzing OpenStreetMap data
import psycopg2                       # For PostgreSQL database connections
from shapely.geometry import Point, Polygon, LineString  # For working with spatial shapes
from shapely.strtree import STRtree   # Fast spatial indexing for geometric queries
import rtree                          # Spatial indexing used for spatial search
import math                           # Math operations like cos, sin, etc.
import pandas as pd                   # Data manipulation and table handling
import requests                       # Making HTTP requests (used for APIs)
from zipfile import ZipFile           # Unzipping downloaded shapefiles
import tempfile                       # Temporary folder to store unzipped files
import shapely                        # Additional geometry operations
from datetime import datetime         # To handle dates and timestamps
import os                             # Access environment variables like API keys
import numpy as np                    # Numerical operations and array handling
import geopandas                      # Spatial data handling built on pandas
import random                         # For generating random samples if needed
from psycopg2 import extras           # For efficient bulk inserts into PostgreSQL
import shapely.geometry as geometry   # Shorthand import for geometry
from pyproj import Transformer        # Coordinate transformation (e.g., from lat/lon to EPSG:3857)
from io import BytesIO                # Used to read files directly from memory

# Importing constants and configuration values for household dataset processing
from household_constants import(
    households_variables_dict,
    households_key_list,
    FIBSCODE,
    YEAR,
    income_ranges,
    size_index_dict,
    workers_index_dict
)

# -----------------------------------------------------------------
# Configuration and Setting of constants for later repitative usage.
# -----------------------------------------------------------------

place_name = "Franklin County, Ohio, USA"   # The geographic area of focus. 

county_code = FIBSCODE[2:]                  # Extract county part of FIPS code
state_code = FIBSCODE[:2]                   # Extract state part of FIPS code

center_point = (39.938806, -82.972361)      # Central coordinate for pulling OSM data
dist = 1000

# Load sensitive credentials from environment variables
PASS = os.getenv("DB_PASS")
APIKEY = os.getenv("APIKEY")
USER = os.getenv("DB_USER")
NAME = os.getenv("DB_NAME")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")

#Starts Census Data Retrieval.  
# Create an empty DataFrame to hold all census data
county_data = pd.DataFrame()

# Since the Census API limits to 50 variables per call, we split the full household variable list into multiple batches of >50
for count in range(int(len(households_key_list)/50)+1):
    variables = ""
    # Create a comma-separated list of variables for the current batch
    if ((count+1)*50) > len(households_key_list):
        variables = ",".join(households_key_list[(50*count):])
    elif count == 0:
        if (int(len(households_key_list)/50)+1) == 1:
            variables = ",".join(households_key_list[:])
        else:
            variables = ",".join(households_key_list[:(50*(count+1)-1)])
    else:
        variables = ",".join(households_key_list[(50*count):(50*(count+1)-1)])

    #Pull data and add to dataframe with constructing the API request URL with the selected variables for this batch.
    url = f"https://api.census.gov/data/{YEAR}/acs/acs5?get=NAME,
    {variables}&for=tract:*&in=state:{state_code}&in=county:{county_code}&key={APIKEY}"

    # Send the request to the Census API and get the response
    response = requests.request("GET", url)
    if len(county_data != 0):
        # Merge the new response with the existing dataset on the 'NAME' field
        county_data = pd.merge(
            pd.DataFrame(response.json()[1:], 
            columns=response.json()[0]), 
            county_data, 
            on='NAME', 
            how='inner')
    else:
         # If this is the first batch, initialize the dataset
        county_data = pd.DataFrame(response.json()[1:], columns=response.json()[0])

# ------------------------------------------
# Load Geographical Tract Data (Shapefiles)
# ------------------------------------------

tract_url = f"https://www2.census.gov/geo/tiger/TIGER{YEAR}/TRACT/tl_{YEAR}_{state_code}_tract.zip"
response = requests.request("GET", tract_url)
# Read the zipped shapefile directly from memory
with ZipFile(BytesIO(response.content)) as zip_ref:
    # Create a temporary directory to extract the zip file
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_ref.extractall(tmpdirname)
        
        # Loop through the extracted files to locate the .shp or .geojson
        for root, dirs, files in os.walk(tmpdirname):
            for file in files:
                if file.endswith(".shp") or file.endswith(".geojson"):
                    file_path = os.path.join(root, file)
                    # Load shapefile/geojson into a GeoDataFrame and convert to EPSG:3857 projection
                    geodata = (geopandas.read_file(file_path)).to_crs("epsg:3857")

# ---------------------------------------------------------------------------
# Merge Geographical Dataframe (containing shapely ploygons) with Census Data
# ---------------------------------------------------------------------------

# Filter down to just the county-level geometries
county_geodata = geodata[geodata['COUNTYFP'] == county_code]

# Rename the tract code column so we can merge it later
county_geodata = county_geodata.rename(columns={"TRACTCE":"tract_y"})

# Convert tract codes to integer for a successful merge
county_geodata["tract_y"] = county_geodata["tract_y"].astype(int)
county_data["tract_y"] = county_data["tract_y"].astype(int)

# Merge geographic and census data on 'tract_y' and # Rename the columns
data = pd.merge(county_geodata, county_data, on = "tract_y", how="inner")
data.rename(columns=households_variables_dict, inplace = True)

# Convert all geometries to EPSG:3857 projection (used for consistent spatial analysis)
data = data.to_crs("epsg:3857")
tract_index = STRtree(data["geometry"])


# ---------------------------------
# Setting up PostgreSQL Connection
# ---------------------------------

# Connect to the PostgreSQL database using credentials from environment variables
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

# Create SQL table to store road metadata and geometry
create_roads_query = '''
CREATE TABLE roads (
    name TEXT,
    highway VARCHAR(30),
    length NUMERIC,
    geometry TEXT,
    service VARCHAR(30)
);
'''

# Define SQL schema for the 'food_stores' table
create_food_stores_query = '''
CREATE TABLE food_stores (
    shop VARCHAR(15),
    geometry TEXT,
    name VARCHAR(50)
);
'''

# Execute command to create the 'roads' table in the database
cursor.execute(create_roads_query)

# Execute command to create the 'food_stores' table in the database
cursor.execute(create_food_stores_query)


# --------------------------------
# Extracting Road Network from OSM
# --------------------------------

place_name = "Franklin County, Ohio, USA"
map_elements = list()  # Holds buffered geometries of interest for later spatial operations
housing_areas = list() # Stores areas near residential roads to simulate housing zones. 

#Get road network from open street maps
G = ox.graph_from_point(center_point,dist=dist, network_type='all',retain_all=True)
# Convert the OSM road graph into GeoDataFrames: one for nodes, one for edges
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

# Convert the coordinate reference system to Web Mercator (meters)
gdf_edges = gdf_edges.to_crs("epsg:3857")

# Depending on available columns, select relevant fields
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

# Prepare road geometry and attributes for insertion into database
gdf_edges["length"] = gdf_edges["length"].astype(int)
gdf_edges["geometry"] = gdf_edges["geometry"].astype(str)
roads_query = "INSERT INTO roads (name,highway,length,geometry,service) VALUES %s"
data_tuples = list(gdf_edges.itertuples(index=False, name=None))


# -------------------------------------
# Extract Food Store Locations from OSM
# -------------------------------------

# Use OSM to fetch food-related retail locations (e.g., supermarkets, groceries)
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

# Convert coordinates to metric projection for spatial operations
features = features.to_crs("epsg:3857")

# Keep only relevant columns for database storage
features = features[["shop","geometry","name"]]

# Insert food stores into PostgreSQL database and prepare data for SQL insert
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

# -----------------------------------
# Create Households Table in Database
# -----------------------------------

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

# Drop existing table to ensure a fresh start
cursor.execute('DROP TABLE IF EXISTS households;')

# Create the new households table
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


# ----------------------------------
# Household Generation and Placement
# ----------------------------------

# Initialize lists and spatial index to store household polygons and allow spatial querying
houses = list()
houses_index = rtree.index.Index()
total_count = 0
total_google_pulls = 0
house_tuples = list()
housing_areas_count = 0

# Iterate through each housing area (area around residential roads) to simulate household placement
for housing_area in housing_areas:
    housing_areas_count += 1

    # Display current progress as a percentage
    print(str(round(housing_areas_count / len(housing_areas) * 100)) + "%")

    count = 0

    # Get the boundary points of the polygon to simulate house edges
    exterior_coords = list(housing_area.exterior.coords)

    # Convert boundary into edge segments to distribute houses along each side
    edges = [LineString([exterior_coords[i], exterior_coords[i + 1]])
             for i in range(len(exterior_coords) - 1)]

    # For each edge segment of the polygon, calculate direction and plan house placement
    for edge in edges:
        # Calculate the vector and length of the edge
        length = edge.length
        coord1 = edge.coords[0]
        coord2 = edge.coords[1]
        vector_direction = (coord2[0] - coord1[0], coord2[1] - coord1[1])
        temp = (vector_direction[0])**2 + (vector_direction[1])**2

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

            # If we got here, that means that this house placement is valid, so we add the house to our list
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
            #Get amount of people in each tract at each income levels
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

            #This is stupid - literally just hardcoded
            household_size = random.choices([1,2,3,4,5,6,7],weights=[1,1,1,1,0,0,0])[0]

            num_workers = 0
            if household_size == 1:
                if sum(worker_weights[:2])==0:
                    num_workers = 0
                else:
                    num_workers = random.choices([0,1], weights=worker_weights[:2], k=1)[0]
            if household_size == 2:
                if sum(worker_weights[2:5])==0:
                    num_workers = 0
                else:
                    num_workers = random.choices([0,1,2], weights=worker_weights[2:5], k=1)[0]
            if household_size == 3:
                if sum(worker_weights[5:9])==0:
                    num_workers = 0
                else:
                    num_workers = random.choices([0,1,2,3], weights=worker_weights[5:9], k=1)[0]
            if household_size >= 4:
                if sum(worker_weights[9:])==0:
                    num_workers = 0
                else:
                    num_workers = random.choices([0,1,2,3], weights=worker_weights[9:], k=1)[0]
            
            size_indexes = None
            if household_size<4:
                size_indexes = size_index_dict[household_size]
            else:
                size_indexes = size_index_dict[4]
            workers_indexes = workers_index_dict[num_workers]
            vehicle_combined_weights = None
            if num_workers != 3:
                vehicle_combined_weights = np.array(vehicle_weights[(size_indexes[0]):(size_indexes[1])])+np.array(vehicle_weights[(workers_indexes[0]):(workers_indexes[1])])
            else:
                vehicle_combined_weights = np.array(vehicle_weights[(size_indexes[0]):(size_indexes[1])])+np.array(vehicle_weights[(workers_indexes[0]):])
            vehicles = random.choices([0,1,2,3,4],weights=vehicle_combined_weights)[0]

            # Get nearest store to house
            nearest_store = None
            store_distance = 100000000
            #TODO get nearest SPM and nearest CSPM
            for store in store_tuples:
                store = shapely.wkt.loads(store[1])
                dist = store.distance(house)
                if dist <= store_distance:
                    nearest_store = store
                    store_distance = dist
            
            # Initialize the Google Maps client with your API key
            #gmaps = googlemaps.Client(key=GOOGLEAPIKEY)

            # Define the source CRS and target CRS (e.g., from EPSG:4326 to EPSG:3857)
            source_crs = "EPSG:3857" # WGS84 (lat/lon)
            target_crs = "EPSG:4326" # Web Mercator (meters)

            # Create a transformer object
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

            # Transform the polygon by transforming each coordinate
            house_4326 = [transformer.transform(x, y) for x, y in house.exterior.coords]

            store_4326 = [transformer.transform(x, y) for x, y in nearest_store.exterior.coords]

            # Create a new Shapely polygon with the transformed coordinates
            house_4326 = geometry.Polygon(house_4326)
            store_4326 = geometry.Polygon(store_4326)
            origin = (float(house_4326.centroid.y), float(house_4326.centroid.x))
            destination = (float(store_4326.centroid.y), float(store_4326.centroid.x))

            # Get transport times to the closest store for each house
            """walking_time = gmaps.directions(origin,
                                                destination,
                                                mode="walking",
                                                departure_time=datetime.now())[0]["legs"][0]["duration"]["text"]"""
            walking_time = 0
            """biking_time = gmaps.directions(origin,
                                                destination,
                                                mode="bicycling",
                                                departure_time=datetime.now())[0]["legs"][0]["duration"]["text"]"""
            biking_time = 0
            """transit_time = gmaps.directions(origin,
                                                destination,
                                                mode="transit",
                                                departure_time=datetime.now())[0]["legs"][0]["duration"]["text"]
                                                """
            transit_time = 0
            #TODO Maybe add driving time if vehicles is > 0 
            """driving_time = gmaps.directions(origin,
                                                destination,
                                                mode="driving",
                                                departure_time=datetime.now())[0]["legs"][0]["duration"]["text"]
                                                """
            driving_time = 0
            total_google_pulls += 1
            house_tuples.append((total_count,str(house),income,household_size,vehicles,num_workers,walking_time,biking_time,transit_time,driving_time))
            total_count+=1


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
