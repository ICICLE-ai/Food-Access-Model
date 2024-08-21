import osmnx as ox

place_name = "Franklin County, Ohio, USA"

#Get road network from open street maps
G = ox.graph_from_place(place_name, network_type='all',retain_all=True)
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

#convert to epsg:3857
gdf_edges = gdf_edges.to_crs("epsg:3857")
gdf_edges = gdf_edges[["name","highway","length","geometry","service","landuse"]]

# Save the DataFrame to a CSV file
gdf_edges.to_csv('data/household_creation/roads.csv', index=False)