from pathlib import Path
import sys

## PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def get_project_root():
    return str(ROOT)

def get_data_path(filename):
    return str(ROOT / "test_beds" / "map_and_route_plotter" / "data" / filename)

import numpy as np
import matplotlib.pyplot as plt

from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle

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

## Configure the map data
map_data = [
    [(0,10000), (10000,10000), (9200,9000) , (7600,8500), (6700,7300), (4900,6500), (4300, 5400), (4700, 4500), (6000,4000), (5800,3600), (4200, 3200), (3200,4100), (2000,4500), (1000,4000), (900,3500), (500,2600), (0,2350)],   # Island 1 
    [(10000, 0), (11500,750), (12000, 2000), (11700, 3000), (11000, 3600), (11250, 4250), (12300, 4000), (13000, 3800), (14000, 3000), (14500, 2300), (15000, 1700), (16000, 800), (17500,0)], # Island 2
    [(15500, 10000), (16000, 9000), (18000, 8000), (19000, 7500), (20000, 6000), (20000, 10000)],
    [(5500, 5300), (6000,5000), (6800, 4500), (8000, 5000), (8700, 5500), (9200, 6700), (8000, 7000), (6700, 6300), (6000, 6000)],
    [(15000, 5000), (14000, 5500), (12500, 5000), (14000, 4100), (16000, 2000), (15700, 3700)],
    [(11000, 2000), (10300, 3200), (9000, 1500), (10000, 1000)]
    ]

map = PolygonObstacle(map_data)
# Own ship
own_ship_route_filename = 'own_ship_route.txt'
own_ship_route_name = get_data_path(own_ship_route_filename)
own_north_route, own_east_route = load_waypoints(own_ship_route_name)

# Collect obstacle ship routes
import glob, os
route_folder = get_data_path("")  # base folder
obs_files = sorted(glob.glob(os.path.join(route_folder, "obs_ship_route_*.txt")))

obs_routes = [(load_waypoints(fname)) for fname in obs_files]

# Plot
plt.figure(figsize=(10, 5.5))

# Own ship
plt.scatter(own_east_route, own_north_route, marker='x', color='blue')
plt.plot(own_east_route, own_north_route, linestyle='--', color='blue', label='own ship')

# Obstacle ships
colors = ["red", "green", "orange", "purple", "brown"]
labels = ["obs ship 1", "obs ship 2", "obs ship 3", "obs ship 4", "obs ship 5"]
for i, (north, east) in enumerate(obs_routes):
    c = colors[i % len(colors)]
    l = labels[i % len(labels)]
    plt.scatter(east, north, marker='x', color=c)
    plt.plot(east, north, linestyle='--', color=c, label=l)

# Map
map.plot_obstacle(plt.gca())

plt.xlim(0, 20000)
plt.ylim(0, 10000)
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