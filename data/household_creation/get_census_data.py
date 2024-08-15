import pandas as pd
import requests
from zipfile import ZipFile
import tempfile
import os
import geopandas
from io import BytesIO
from household_constants import(
    households_variables_dict,
    households_key_list,
    FIBSCODE,
    YEAR
)

county_code = FIBSCODE[2:]
state_code = FIBSCODE[:2]

from config import APIKEY

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