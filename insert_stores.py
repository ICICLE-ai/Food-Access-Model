#!/usr/bin/env python3
"""Insert stores from CSV into database"""

import sys
import psycopg2
from pyproj import Transformer
from shapely import Polygon
import math
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def convert_to_polygon(lat, lon, store_type):
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    lon, lat = transformer.transform(lon, lat)
    
    if store_type in ["supermarket", "grocery", "greengrocer"]:
        polygon = Polygon([(lon + 50 * math.cos(math.radians(angle)), 
                           lat + 50 * math.sin(math.radians(angle))) 
                          for angle in range(0, 360, 60)])
    else:
        polygon = Polygon([(lon, lat + 20), (lon + 25, lat - 30), (lon - 25, lat - 30)])
    
    return polygon.wkt

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python insert_stores.py your_stores.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} stores\n")
    
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM simulation_instances WHERE name = 'default_simulation';")
    sim_id = cursor.fetchone()[0]
    
    cursor.execute("SELECT MAX(store_id) FROM food_stores WHERE simulation_instance = %s AND simulation_step = 0", (sim_id,))
    max_id = cursor.fetchone()[0] or 0
    
    for idx, row in df.iterrows():
        store_name = str(row['Name'])[:50]  # Truncate to 50 chars
        store_type = str(row['Type'])[:15]  # Truncate to 15 chars
        # CSV has lat/long swapped - lat column has longitude values, long column has latitude values
        geometry = convert_to_polygon(float(row['long']), float(row['lat']), store_type)
        cursor.execute("""
            INSERT INTO food_stores (simulation_instance, simulation_step, shop, geometry, name, store_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (sim_id, 0, store_type, geometry, store_name, max_id + idx + 1))
        print(f"✓ {store_name}")
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"\nInserted {len(df)} stores!")
