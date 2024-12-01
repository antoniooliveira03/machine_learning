import json
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, Polygon, MultiPolygon

def map_(geojson_file, df):
    """
    This function takes a GeoJSON file and a DataFrame, matches the counties from the DataFrame
    with the counties in the GeoJSON, and plots the geometries on a map with a basemap.

    Args:
        geojson_file (str): Path to the GeoJSON file containing county geometries.
        df (pandas.DataFrame): DataFrame containing the 'County of Injury' column.
    """
    # Load the GeoJSON data
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)

    # Extract the county names and coordinates from the GeoJSON
    county_mapping = {}
    for feature in geojson_data["features"]:
        county_name = feature["properties"]["NAME"].upper()  # Convert to uppercase for matching
        county_mapping[county_name] = feature["geometry"]  # Store geometry for later extraction

    # Now match the County of Injury values to the counties in the GeoJSON
    injury_counties = df['County of Injury'].dropna().unique()

    non_matched = {}
    matched_counties = {}
    matched_coordinates = {}

    for county in injury_counties:
        county_upper = county.upper()  # Convert to uppercase to match
        if county_upper in county_mapping:
            matched_counties[county_upper] = county_mapping[county_upper]
            # Extract the coordinates for the matched county
            geometry = county_mapping[county_upper]
            if geometry["type"] == "Point":
                # For point geometries, extract the single coordinate
                matched_coordinates[county_upper] = geometry["coordinates"]
            elif geometry["type"] == "Polygon" or geometry["type"] == "MultiPolygon":
                # For polygons and multipolygons, extract the coordinates
                matched_coordinates[county_upper] = geometry["coordinates"]
        else:
            non_matched[county_upper] = "No match found"  # Store a message for unmatched counties

    # Check if there are any matched coordinates to plot
    if not matched_coordinates:
        print("No matched coordinates to plot!")
        return

    # Create an empty list to store the geometries
    geometries = []

    # Loop through the matched coordinates and handle geometry correctly
    for county, coords in matched_coordinates.items():
        try:
            # Handle Point (latitude, longitude) as a simple point geometry
            if isinstance(coords, list) and len(coords) == 1 and isinstance(coords[0], float):  
                # If it's a float in a list, it's a point (lat, lon)
                geometries.append(Point(coords[0]))  # Assuming coords are in (lat, lon) format

            # Handle Polygon or MultiPolygon geometries
            elif isinstance(coords, list):  # If coords is a list, it could be a Polygon or MultiPolygon
                if isinstance(coords[0], list):  # If the first element is a list of coordinates (for polygons)
                    if len(coords) == 1:  # Single polygon
                        geometries.append(Polygon(coords[0]))  # polygon[0] is the boundary of the polygon
                    else:  # MultiPolygon
                        multipolygon = MultiPolygon([Polygon(p[0]) for p in coords])
                        geometries.append(multipolygon)
                else:  # If it's neither a list of coordinates, handle it differently
                    raise ValueError(f"Invalid format for {county} with coordinates: {coords}")

            else:
                raise ValueError(f"Unexpected coordinate format for {county}: {coords}")

        except Exception as e:
            # Log the error and the county name
            print(f"Error processing {county}: {e}")

    # Check if there are any empty geometries
    if not geometries:
        print("No geometries were successfully processed!")
        return

    # Create a GeoDataFrame from the geometries
    gdf = gpd.GeoDataFrame(geometry=geometries)

    # Set CRS (Coordinate Reference System) for the GeoDataFrame to match the basemap
    gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)

    # Plotting with background map
    ax = gdf.plot(figsize=(10, 10), alpha=0.7, edgecolor='k', color='orange')

    # Add basemap using contextily (Web Map Tile Service)
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    # Show plot
    plt.show()