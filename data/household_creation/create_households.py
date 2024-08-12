import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Polygon, Point, mapping
import shapely
import random
import requests
from io import BytesIO
from zipfile import ZipFile
import tempfile
import os
import pyproj
import rasterio
import pyproj
from rasterio.mask import mask
from household_constants import(
    households_variables_dict,
    household_values_list,
    households_key_list,
    income_ranges,
    size_index_dict,
    workers_index_dict
)

FIBSCODE = "39049"
YEAR = 2022
from config import APIKEY

county_code = FIBSCODE[2:]
state_code = FIBSCODE[:2]

# Open the raster file and read the first band
with rasterio.open('data/household_creation/county_raster.tif') as src:
    band1 = src.read(1)  # Read the first band
    raster_crs = src.crs  # Get the CRS of the raster
    transform_affline = src.transform # Get the affine transformation of the raster

# Function to generate a random point within a polygon
def place_household(tract_polygon,polygons):
    min_x, min_y, max_x, max_y = tract_polygon.bounds
    count = 0
    while True:
        # Generate a random point
        location = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        # Check if the point is inside the polygon
        
        if tract_polygon.contains(location):
            count += 1
            if count == 10000:
                raise Exception()
            polygon =Polygon(((location.x+20, location.y+20),
                              (location.x, location.y+40),
                              (location.x-20, location.y+20),
                              (location.x+20, location.y+20),
                              (location.x-20, location.y+20),
                              (location.x-20, location.y-10),
                              (location.x-5, location.y-10),
                              (location.x-5, location.y+5),
                              (location.x+5, location.y+5),
                              (location.x+5, location.y-10),
                              (location.x-5, location.y-10),
                              (location.x+20, location.y-10)))
            not_touching = True
            for polygon_2 in polygons:
                
                touches = polygon.intersects(polygon_2)
                if touches:
                    not_touching = False
                    break

            nlcd_transformer = pyproj.Transformer.from_crs(raster_crs, 'EPSG:3857')

            in_housing_area = True
            points = [
                (location.x+20, location.y+20),
                (location.x, location.y+40),
                (location.x-20, location.y+20),
                (location.x+20, location.y+20),
                (location.x-20, location.y+20),
                (location.x-20, location.y-10),
                (location.x-5, location.y-10),
                (location.x-5, location.y+5),
                (location.x+5, location.y+5),
                (location.x+5, location.y-10),
                (location.x-5, location.y-10),
                (location.x+20, location.y-10)
            ]
            for i in range(len(points)):
                # Transform the point from EPSG:3857 to the raster's CRS
                x_raster, y_raster = nlcd_transformer.transform(points[i][0], points[i][1])

                # Convert the transformed coordinates to row and column indices
                row, col = ~transform_affline * (x_raster, y_raster)

                # Convert to integers (row and column indices must be integers)
                row = int(row)
                col = int(col)

                # Extract the value from the NumPy array at the calculated indices
                value = band1[col, row]
                if ((value != 24) and (value != 23) and (value != 25)):
                    in_housing_area = False
            if not_touching and in_housing_area:
                return polygon


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
                    geodata = geopandas.read_file(file_path)


#Merge geographical dataframe (containing shapely ploygons) with census data
geodata.crs = 'EPSG:3857'
county_geodata = geodata[geodata['COUNTYFP'] == county_code]
county_geodata = county_geodata.rename(columns={"TRACTCE":"tract_y"})
county_geodata["tract_y"] = county_geodata["tract_y"].astype(int)
county_data["tract_y"] = county_data["tract_y"].astype(int)
data = pd.merge(county_geodata, county_data, on = "tract_y", how="inner")
data.rename(columns=households_variables_dict, inplace = True)
households = pd.DataFrame(columns = ["id","polygon","income","household_size","vehicles","number_of_workers"])

#helper method to switch x and y in a shapely Point
def swap_xy(x, y):
    return y, x

# Create a list of all store polygons so that we can test if households overlap with them
store_polygons = []
stores = pd.read_csv("data/stores.csv")
for index,row in stores.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]
    point = Point(lat,lon)
    project = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:4326'), # source coordinate system
        pyproj.Proj('epsg:3857')) # destination coordinate system
    point = shapely.ops.transform(project.transform, point)  # apply projection
    polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
    store_polygons.append(polygon)  # apply projection


#Iterate through each tract and create households
total_count = 0
for index,row in data.iterrows():
    if ((row['tract_y']>5000)&(row['tract_y']<6000)):
        #Create household polygon
        tract_polygon = Polygon(row["geometry"])
        tract_polygon = shapely.ops.transform(swap_xy, tract_polygon)
        project = pyproj.Transformer.from_proj(
            pyproj.Proj('epsg:4326'), # source coordinate system
            pyproj.Proj('epsg:3857')) # destination coordinate system
        tract_polygon = shapely.ops.transform(project.transform, tract_polygon)  # apply

        #Get amount of people in each tract at each income level
        income_weights = np.array(row["10k to 15k":"150k to 200k"]).astype(int)
        if sum(income_weights)==0:
            continue

        # Make a list of incomes to distribute to households
        total_households = int(row["total households in tract"])
        distributed_incomes = []
        for i in range(14):
            uniform_list = []
            if i != 14:
                uniform_list = np.random.uniform(income_ranges[i][0],income_ranges[i][1],income_weights[i])
            else:
                uniform_list = np.random.uniform(200000,200000,income_weights[i])
            distributed_incomes.extend(uniform_list.astype(int))

        vehicle_weights = [
                            int(row["1 Person(s) 0 Vehicle(s)"]),
                            int(row["1 Person(s) 1 Vehicle(s)"]),
                            int(row["1 Person(s) 2 Vehicle(s)"]),
                            int(row["1 Person(s) 3 Vehicle(s)"]),
                            int(row["1 Person(s) 4+ Vehicle(s)"]),
                            int(row["2 Person(s) 0 Vehicle(s)"]),
                            int(row["2 Person(s) 1 Vehicle(s)"]),
                            int(row["2 Person(s) 2 Vehicle(s)"]),
                            int(row["2 Person(s) 3 Vehicle(s)"]),
                            int(row["2 Person(s) 4+ Vehicle(s)"]),
                            int(row["3 Person(s) 0 Vehicle(s)"]),
                            int(row["3 Person(s) 1 Vehicle(s)"]),
                            int(row["3 Person(s) 2 Vehicle(s)"]),
                            int(row["3 Person(s) 3 Vehicle(s)"]),
                            int(row["3 Person(s) 4+ Vehicle(s)"]),
                            int(row["4+ Person(s) 0 Vehicle(s)"]),
                            int(row["4+ Person(s) 1 Vehicle(s)"]),
                            int(row["4+ Person(s) 2 Vehicle(s)"]),
                            int(row["4+ Person(s) 3 Vehicle(s)"]),
                            int(row["4+ Person(s) 4+ Vehicle(s)"]),
                            int(row["0 Worker(s) 0 Vehicle(s)"]),
                            int(row["0 Worker(s) 1 Vehicle(s)"]),
                            int(row["0 Worker(s) 2 Vehicle(s)"]),
                            int(row["0 Worker(s) 3 Vehicle(s)"]),
                            int(row["0 Worker(s) 4+ Vehicle(s)"]),
                            int(row["1 Worker(s) 0 Vehicle(s)"]),
                            int(row["1 Worker(s) 1 Vehicle(s)"]),
                            int(row["1 Worker(s) 2 Vehicle(s)"]),
                            int(row["1 Worker(s) 3 Vehicle(s)"]),
                            int(row["1 Worker(s) 4+ Vehicle(s)"]),
                            int(row["2 Worker(s) 0 Vehicle(s)"]),
                            int(row["2 Worker(s) 1 Vehicle(s)"]),
                            int(row["2 Worker(s) 2 Vehicle(s)"]),
                            int(row["2 Worker(s) 3 Vehicle(s)"]),
                            int(row["2 Worker(s) 4+ Vehicle(s)"]),
                            int(row["3+ Worker(s) 0 Vehicle(s)"]),
                            int(row["3+ Worker(s) 1 Vehicle(s)"]),
                            int(row["3+ Worker(s) 2 Vehicle(s)"]),
                            int(row["3+ Worker(s) 3 Vehicle(s)"]),
                            int(row["3+ Worker(s) 4+ Vehicle(s)"])
                        ]
        vehicle_weights = [0 if item == -666666666 else item for item in vehicle_weights]

        worker_weights = [
                            int(row["1 Person(s) 0 Worker(s)"]),
                            int(row["1 Person(s) 1 Worker(s)"]),
                            int(row["2 Person(s) 0 Worker(s)"]),
                            int(row["2 Person(s) 1 Worker(s)"]),
                            int(row["2 Person(s) 2 Worker(s)"]),
                            int(row["3 Person(s) 0 Worker(s)"]),
                            int(row["3 Person(s) 1 Worker(s)"]),
                            int(row["3 Person(s) 2 Worker(s)"]),
                            int(row["3 Person(s) 3 Worker(s)"]),
                            int(row["4+ Person(s) 0 Worker(s)"]),
                            int(row["4+ Person(s) 1 Worker(s)"]),
                            int(row["4+ Person(s) 2 Worker(s)"]),
                            int(row["4+ Person(s) 3+ Worker(s)"]),
                        ]
        worker_weights = [0 if item == -666666666 else item for item in worker_weights]

        household_size_weights = [
                            int(row["Median Income for 1 Person(s)"]),
                            int(row["Median Income for 2 Person(s)"]),
                            int(row["Median Income for 3 Person(s)"]),
                            int(row["Median Income for 4 Person(s)"]),
                            int(row["Median Income for 5 Person(s)"]),
                            int(row["Median Income for 6 Person(s)"]),
                            int(row["Median Income for 7+ Person(s)"])
                        ]
        household_size_weights = [0 if item == -666666666 else item for item in household_size_weights]

        polygons = store_polygons
        geojson_polygon = [mapping(tract_polygon)]
        with rasterio.open('data/household_creation/county_raster.tif') as src:
            band1 = src.read(1)  # Read the first band
            raster_crs = src.crs  # Get the CRS of the raster
            transform_affline = src.transform # Get the affine transformation of the raster
            out_image, out_transform = mask(src, shapes=geojson_polygon, crop=True)

        total_avail_area = 0
        for num in np.nditer(out_image):
            if ((num == 24) or (num == 23) or (num == 25)):
                total_avail_area += 1
        num_houses = int(total_avail_area/5)
        print(num_houses)

        for household_num in range(num_houses):

            location = Point()
            polygon = Polygon()
            polygon = place_household(tract_polygon,polygons)

            income_range = random.choices(income_ranges,weights = income_weights)
            income = random.randint(int(income_range[0][0]/1000),int(income_range[0][1]/1000))*1000

            #This is stupid - literally just hardcoded
            household_size = random.choices([1,2,3,4,5,6,7],weights=[1,1,1,1,0,0,0])[0]

            num_workers = 0
            if household_size == 1:
                num_workers = random.choices([0,1], weights=worker_weights[:2], k=1)[0]
            if household_size == 2:
                num_workers = random.choices([0,1,2], weights=worker_weights[2:5], k=1)[0]
            if household_size == 3:
                num_workers = random.choices([0,1,2,3], weights=worker_weights[5:9], k=1)[0]
            if household_size >= 4:
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

            households.loc[total_count] = {
                "id":total_count,
                "polygon":polygon,
                "income":income,
                "vehicles":vehicles,
                "household_size":household_size,
                "number_of_workers":num_workers
                }
            total_count+=1
            polygons.append(polygon)

households.to_csv('data/households.csv', index=False)
print(households)