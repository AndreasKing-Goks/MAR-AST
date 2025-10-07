from pathlib import Path
import sys

## PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from utils.get_path import get_ship_route_path, get_map_path
from utils.prepare_map import get_gdf_from_gpkg

def load_waypoints(route, print_init_msg=False):
    ''' Reads the file containing the route and stores it as an
        array of north positions and an array of east positions
    '''
    # self.data = np.loadtxt(route)
    # self.data = route
    if print_init_msg:
        print(f"Route received in load_waypoints: {route}")
    
    # Load the file if the input is a string (file path)
    if isinstance(route, str):
        if print_init_msg:
            print(f"Loading route file from: {route}")  # Debugging
        data = np.loadtxt(route)
    else:
        data = route  # Assume it's already an array
    
    north = []
    east = []
    for i in range(0, (int(np.size(data) / 2))):
        north.append(data[i][0])
        east.append(data[i][1])
        
    return north, east

# Own ship
own_ship_route_filename = 'own_ship_route.txt'
own_ship_route_name = get_ship_route_path(ROOT, own_ship_route_filename)
own_north_route, own_east_route = load_waypoints(own_ship_route_name)

# Collect obstacle ship routes
import glob, os
route_folder = get_ship_route_path(ROOT, "")  # base folder
tar_files = sorted(glob.glob(os.path.join(route_folder, "tar_ship_route_*.txt")))

tar_routes = [(load_waypoints(fname)) for fname in tar_files]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

## Map
# -----------------------
# GPKG settings (edit if your layer names differ)
# -----------------------
GPKG_PATH   = get_map_path(ROOT, "basemap.gpkg")       # <-- put your file here (or absolute path)
FRAME_LAYER = "frame_3857"
OCEAN_LAYER = "ocean_3857"
LAND_LAYER  = "land_3857"
COAST_LAYER = "coast_3857"               # optional
WATER_LAYER = "water_3857"               # optional

frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf = get_gdf_from_gpkg(GPKG_PATH, FRAME_LAYER, OCEAN_LAYER, LAND_LAYER, COAST_LAYER, WATER_LAYER)

# draw order: land first, then ocean/water, then coast lines, then frame boundary
if not land_gdf.empty:
    land_gdf.plot(ax=ax, facecolor="#e8e4d8", edgecolor="#b5b2a6", linewidth=0.4, zorder=1)
if not ocean_gdf.empty:
    ocean_gdf.plot(ax=ax, facecolor="#d9f2ff", edgecolor="#bde9ff", linewidth=0.4, alpha=0.95, zorder=2)
if not water_gdf.empty:
    water_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="#74a8d8", linewidth=0.4, alpha=0.95, zorder=2)
if not coast_gdf.empty:
    coast_gdf.plot(ax=ax, color="#2f7f3f", linewidth=1.0, zorder=3)

# fit exactly to frame bounds, remove margins/axes so the basemap fills the figure
minx, miny, maxx, maxy = frame_gdf.total_bounds
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
ax.set_aspect("equal"); ax.set_axis_off(); ax.margins(0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Own ship
plt.scatter(own_east_route, own_north_route, marker='x', color='blue')
plt.plot(own_east_route, own_north_route, linestyle='--', color='blue', label='own ship')

# Obstacle ships
colors = ["red", "green", "orange", "purple", "brown"]
labels = ["target ship 1", "target ship 2", "target ship 3", "target ship 4", "target ship 5"]
for i, (north, east) in enumerate(tar_routes):
    c = colors[i % len(colors)]
    l = labels[i % len(labels)]
    plt.scatter(east, north, marker='x', color=c)
    plt.plot(east, north, linestyle='--', color=c, label=l)

plt.title('Map and Route Plot')
plt.xlabel('East position (m)')
plt.ylabel('North position (m)')
plt.gca().set_aspect('equal')
plt.grid(color='0.8', linestyle='-', linewidth=0.5)
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Show Plot
plt.show()