# MAR-AST
Repository for Adaptive Stress Testing (AST) algorithm for maritime operations

## Conda Environment Setup [Simulator Only]

First clone the repository. Made sure conda is installed. Then, set up the conda environment by running this command in terminal:

```bash
conda env create -f mar-ast.yml
```

## Dependencies for Adaptive Stress Testing
### PyTorch
First we want to install `pytorch`. We use `pytorch` to build the multi-layer perceptron network related with Reinforcement Learning (RL) implementaion for the AST. Please consult to [PyTorch official website](https://pytorch.org/get-started/locally/) for local installation guide. This model is developed on Windows OS, using CUDA 12.6-compliant NVIDA GPU. Install this package by running:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For non-NVIDIA GPU device, use
```bash
pip3 install torch torchvision
```

### Gymnasium
We also use `gymnasium` to define the RL observation and action space. Install this package by running:
```bash
pip install gymnasium
```

### Stable-Baselines 3
`stable-baselines3` is a `pytorch`-compliant version for RL algorithms implementation. This package will be used as the core for our AST algorithm. Install this package by running:
```bash
pip install 'stable-baselines3[extra]'
```

---

##  Improved Ship in Transit Simulator

The **ship-in-transit-simulator** is a modular Python-based simulation framework for modeling and running transit scenarios of a marine vessel. It includes ship dynamics, machinery system behaviors, navigation logic, and environmental effects. 

This simulator is developed based on Ship in Transit Simulator created by Børge Rokseth (**borge.rokseth@ntnu.no**). Original simulator can be found [here](https://github.com/BorgeRokseth/ship_in_transit_simulator.git).


### Ship Dynamics

The ship model simulates motion in **three degrees of freedom**:

- **Surge**
- **Sway**
- **Yaw**

#### State Variables

The full system state includes:

- `x_N`: North position [m]
- `y_E`: East position [m]
- `ψ`: Yaw angle [rad]
- `u`: Surge velocity [m/s]
- `v`: Sway velocity [m/s]
- `r`: Yaw rate (turn rate) [rad/s]

> ⚠️ **ANGLE IN NED FRAME**
>
> When we first initiate the ship heading, `0°` starts from `NORTH (y+)` direction. **Positive heading** is clockwise in direction.


**Additional states** (depending on machinery system model):

- `ω_prop`: Propeller shaft angular velocity [rad/s] — for detailed machinery system
- `T`: Thrust force [N] — for simplified model

#### Forces Modeled

- Inertial forces  
- Added mass effects  
- Coriolis forces  
- Linear and nonlinear damping  
- Environmental forces (wave, wind and current)  
- Control forces (propeller & rudder)

### Machinery System

The ship includes:
- **1 propeller shaft**
- **1 rudder**
- Powered by either a **main diesel engine** or a **hybrid shaft generator**.

#### Energy Sources

- **Main Engine (ME)**: Primary mechanical power source  
- **Hybrid Shaft Generator (HSG)**: Can operate as motor/generator  
- **Electrical Distribution**: Powered by diesel generators

#### Machinery Modes

| Mode   | Propulsion Power      | Hotel Load Power       |
|--------|------------------------|-------------------------|
| PTO    | Main Engine            | Hybrid SG (generator)   |
| MEC    | Main Engine            | Electrical Distribution |
| PTI    | Hybrid SG (motor)      | Electrical Distribution |

#### Available Machinery Models

1. **Detailed Model** — includes propeller shaft dynamics  
2. **Simplified Model** — thrust modeled as a 2nd order transfer function  

### Navigation System

Two navigation modes:

- **Heading-by-Reference**: Maintains a specified heading.  
- **Waypoint Controller**: Follows a sequence of waypoints using Line of Sight (LOS) guidance.

### Environmental Forces

- **Wave** : Based from JONSWAP Spectrum and Spreading Function. Stochasticity is originated from random phases and advanced in time.  
- **Wind** : Based from NORSOK Spectrum. Stochasticity is drawn from Ornstein-Uhlenbeck process.  
- **Current** : Stochasticity is drawn from Ornstein-Uhlenbeck process.  

### Speed Control Options

1. **Throttle Input**: A direct command representing propulsion load percentage  
2. **Speed Controller**: Regulates propeller shaft speed to maintain desired vessel speed  

###  Example Scenarios

- Single-engine configuration using only the Main Engine in PTO mode  
- Complex hybrid-electric propulsion control  

---

## Importing Route and Map from Open Street Map

The repository provides utilities for importing and visualizing **real-world maps** from [OpenStreetMap (OSM)](https://www.openstreetmap.org/). These are primarily used to define simulation areas and plan ship routes within realistic maritime environments.

### Overview

The **map import system** is designed to integrate OSM-based geographic data with the ship simulator.  
Using this, users can:
- Load real coastal or harbor maps as simulation backdrops.  
- Define navigable areas and obstacles based on OSM geometry.  
- Combine OSM-derived maps with manually added obstacles for detailed scenario definition.

### Workflow

1. **Prepare the Map File**
   - The repository provides tools to extract OSM data and save it as a `.gpkg` (GeoPackage) file.  
   - Each layer in the `.gpkg` file (e.g., “frame”, “obstacles”, “routes”) corresponds to different spatial elements used by the simulator.

2. **Load Map in the Simulator**
   - Use helper functions from `utils.prepare_map` such as:
     ```python
     from utils.prepare_map import get_gdf_from_gpkg
     frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf = get_gdf_from_gpkg(GPKG_PATH, FRAME_LAYER, OCEAN_LAYER, LAND_LAYER, COAST_LAYER, WATER_LAYER)
     ```
   - The map geometry is read using `geopandas` and can be directly visualized or used in the environment setup.

3. **Define and Visualize Routes**
   - Route files (typically in `.txt`) define the waypoints of the ship’s mission.  
   - These can be plotted together with the map using:
     ```python
     from utils.get_path import get_map_path, get_ship_route_path
     ```
   - Visualization scripts are provided in `test_beds/map_and_route_plotter/plot_realmap_route.py`.

4. **Typical Use Case**
   - Extract the harbor region from OSM.  
   - Generate a `.gpkg` file containing relevant boundaries and water areas.  
   - Load the map into the simulator and overlay route waypoints to verify the navigation corridor.

5. **File Structure Example**
   ```
   data/
   ├── map/
   │   └── basemap.gpkg
   └── route/
       └── own_ship_route.txt
   ```

6. **Supported Layers**
   The imported `.gpkg` may include:
   - **Frame**: Simulation area boundary  
   - **Water**: Navigable regions  
   - **Obstacle**: Islands, piers, or restricted areas  
   - **Route**: Reference paths for the ship

### Example Script

To visualize both the map and the route:
```bash
python test_beds/map_and_route_plotter/plot_realmap_route.py
```

This script loads the `.gpkg` map and overlays all route files found in `data/route`.  
The result is a clear view of how your simulation environment aligns with real-world geography.

---

## Setting up the Ship in Transit Simulator
For usage and integration examples, refer to the provided scripts in `test_beds`.

Generally, building this simulator is done by doing these steps:
1. Prepare configuration objects to build one ship asset:
   - Ship configurations built from `ShipConfiguration()`
   - Machinery configurations (including machinery modes) built from `MachinerySystemConfiguration()`
   - Simulator configuration built from `SimulationConfiguration()`
2. Using these configuration objects, build the ship using `ShipModel()`.
3. The ship model needs throttle and heading controllers to carry out an autonomous mission.  
   - A simple engine throttle controller with fixed desired speed can be set up using `EngineThrottleFromSpeedSetPoint()`.  
   - A heading controller using LOS Guidance can be set up using `HeadingByRouteController()`. For this, mission waypoints should be provided in a file named `_ship_route.txt`.
4. To allow a ship to “choose” new intermediate waypoints during simulation, use `HeadingBySampledRouteController()`.  
   This builds on the LOS controller but only requires two initial waypoints (start and end) in `_ship_route.txt`.
5. Multiple ship models can run simultaneously. For this, use `ShipAssets()` — a collection of `ShipModel()` objects and associated parameters. Typically:
   - The **first** ship asset represents the **own** ship (under test)
   - The **remaining** ships act as **target** or **obstacle** ships
6. These assets are passed to the RL environment for adaptive stress testing.

---

## Test Beds

In `test_beds`, we provide several example scripts to better understand how the simulator and environment work:

### `env_load_model`
Scripts for testing environmental models.  
See [here](https://github.com/AndreasKing-Goks/MAR-AST/tree/main/test_beds/env_load_model).

| Script | Description |
|--------|-------------|
| `current_model_test.py` | Test the current model using `current_model.py` subsystem  |
| `wave_model_test.py` | Test the wave model using `wave_model.py` subsystem |
| `wind_model_test.py` | Test the wind model using `wind_model.py` subsystem |

### `map_and_route_plotter`
Scripts for placing maps and ship routes for visualization — useful for arranging route waypoints before simulation.  
See [here](https://github.com/AndreasKing-Goks/MAR-AST/tree/main/test_beds/map_and_route_plotter).

| Script | Description |
|--------|-------------|
| `plot_map_route.py` | Plot routes stored in `test_beds/map_route_plotter/data` with maps manually designed using `PolygonObstacle()` class. |
| `plot_realmap_route.py` | Plot routes stored in `data/route` with real-world maps retrieved from [Open Street Map (OSM)](https://www.openstreetmap.org/). |

### `ship_simu_test`
Scripts for using the environment classes from `env_wrappers`.  
See [here](https://github.com/AndreasKing-Goks/MAR-AST/tree/main/test_beds).

| Script | Description |
|--------|-------------|
| `test_double_ship_map.py` | Run `MultiShipEnv()` class on a real-world map while simulating two ship asset. Used to check the COLAV system. |
| `test_multi_ship_map.py` | Run `MultiShipEnv()` class on a real-world map with multiple ship asset simulated together. |
| `test_multi_ship.py` | Run `MultiShipEnv()` class on a simple map with multiple ship asset simulated together. |
| `test_single_ship_map.py` | Run `MultiShipEnv()` class on a real-world map while simulating single ship asset. |
| `test_single_ship.py` | Run `SingleShipEnv()` class for single-ship cases. |