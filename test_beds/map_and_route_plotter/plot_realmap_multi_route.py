from pathlib import Path
import sys

## PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from utils.get_path import get_ship_route_path_for_training, get_map_path
from utils.prepare_map import get_gdf_from_gpkg

# ---------- PATH HELPERS ----------
def load_waypoints(route_path, print_init_msg=False):
    """Route file: two columns [north, east]. Accepts str or Path or array."""
    if print_init_msg:
        print(f"Route received in load_waypoints: {route_path}")
    if isinstance(route_path, (str, Path)):
        data = np.loadtxt(route_path)
    else:
        data = route_path
    north = data[:, 0].tolist()
    east  = data[:, 1].tolist()
    return north, east

# ---------- COLLECT ROUTES ----------
# (A) all files in for_training
route_files = get_ship_route_path_for_training(ROOT, "*", pattern="*.txt")
# e.g., ['1.txt','2.txt',...]; each will become one line on the plot.

routes = [load_waypoints(p) for p in route_files]  # [(north, east), ...]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

## Map
# -----------------------
# GPKG settings (edit if your layer names differ)
# -----------------------
GPKG_PATH   = get_map_path(ROOT, "Stangvik.gpkg")       # <-- put your file here (or absolute path)
FRAME_LAYER = "frame_3857"
OCEAN_LAYER = "ocean_3857"
LAND_LAYER  = "land_3857"
COAST_LAYER = "coast_3857"               # optional
WATER_LAYER = "water_3857"               # optional

frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf = get_gdf_from_gpkg(GPKG_PATH, FRAME_LAYER, OCEAN_LAYER, LAND_LAYER, COAST_LAYER, WATER_LAYER)

# draw order: land first, then ocean/water, then coast lines, then frame boundary
if not land_gdf.empty:
    land_gdf.plot(ax=ax, facecolor="#e6e6e6", edgecolor="#bbbbbb", linewidth=0.3, zorder=0)
if not ocean_gdf.empty:
    ocean_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="none", zorder=1)
if not water_gdf.empty:
    water_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="#74a8d8", linewidth=0.4, alpha=0.95, zorder=2)
if not coast_gdf.empty:
    coast_gdf.plot(ax=ax, color="#2f7f3f", linewidth=1.2, zorder=3)

# fit exactly to frame bounds, remove margins/axes so the basemap fills the figure
minx, miny, maxx, maxy = frame_gdf.total_bounds
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
ax.set_aspect("equal"); ax.set_axis_off(); ax.margins(0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Obstacle ships
for north, east in routes:
    plt.scatter(east, north, marker='x')
    plt.plot(east, north, linestyle='--')

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