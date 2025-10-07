# MAR-AST
Repository for Adaptive Stress Testing (AST) algorithm for maritime operations

## Conda Environment Setup

First clone the repository. Made sure conda is installed. Then, set up the conda environment by running this command in terminal:

```bash
conda env create -f mar-ast.yml
```
---

##  Improved Ship in Transit Simulator

The **ship-in-transit-simulator** is a modular Python-based simulation framework for modeling and running transit scenarios of a marine vessel. It includes ship dynamics, machinery system behaviors, navigation logic, and environmental effects. 

This simulator is developed based on Ship in Transit Simulator created by Børge Rokseth (**borge.rokseth@ntnu.no**). Original simulator can be found at: https://github.com/BorgeRokseth/ship_in_transit_simulator.git


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

- **Wave**      : Based from JONSWAP Spectrum and Spreading Function. Stochasticity is originated from random phases and advanced in time.
- **Wind**      : Based from NORSOK Spectrum. Stochasticity is drawn from Ornstein-Uhlenbeck process.
- **Current**   : Stochasticity is drawn from Ornstein-Uhlenbeck process.

### Speed Control Options

1. **Throttle Input**: A direct command representing propulsion load percentage
2. **Speed Controller**: Regulates propeller shaft speed to maintain desired vessel speed

###  Example Scenarios

- Single-engine configuration using only the Main Engine in PTO mode
- Complex hybrid-electric propulsion control

---

## Setting up the Ship in Transit Simulator
For usage and integration examples, refer to the provided scripts in `test_beds`.

Generally, building this simulator is done by doing these steps:
1. Prepare configurations objects to build one ship asset:
  - Ship configurations built from `ShipConfiguration()`
  - Machinery configurations (including machinery modes) built from `MachinerySystemConfiguration()`
  - Simulator configuration built from `SimulationConfiguration()`
2. Using this configuration objects, we can then built a ship for using `ShipModel()`.
3. The ship model needs throttle controller and heading controller for it to carry an autonomous mission. A simple engine throttle controller with fixed desired speed can be set up using `EngineThrottleFromSpeedSetPoint()`. A heading controller using LOS Guidance can also be set using `HeadingByRouteController()`. For this we need the mission waypoints inside a file named `_ship_route.txt` beforehand.
4. We could also set up a ship model in which it can "choose" a new intermediate waypoint during the simulation. This can be done using `HeadingBySampledRouteController()`. This control is still based on the `HeadingByRouteController()`. The main difference is that the we only need two initial waypoints, start and end point, in the `_ship_route.txt`.
5. We can run multiple ship models simultaneously. In order to do that, we know introduce a new term `ShipAssets()`, where it collects the `ShipModel()` class  along with other parameters associated to each ship. Typicially in stress testing settings, we have a ship asset which undergoes a stress testing, namely `own` ship, and a ship asset that acts as a disturbance to affect the ship under test, namely `target` or `tar`. All ship assets is collected inside a list, then used to create the RL environment for the adaptive stress testing.
6. Typically, the first entry in the ship asset list are the `own` ship, and the rest is `tar` ship/s. 

> Multiple obstacle ships implementation is still unfinished!

---
## Test Beds

In `test_beds`, we provided several examples that can be run to better understand how the code works. Below are the descriptions:

### `env_load_model`
Consists of scripts for running the environmental models. Go [here](https://github.com/AndreasKing-Goks/MAR-AST/tree/main/test_beds/env_load_model).

| Script Name | Description |
|----------|-------------|
| `current_model_test.py` | Test the current model using `current_model.py` subsystem  |
| `wave_model_test.py` | Test the wave model using `wave_model.py` subsystem |
| `wind_model_test.py` | Test the wind model using `wind_model.py` subsystem |


### `map_and_route_plotter`
Consists of scripts for placing the map and ship routes for plotting. Useful to arrange the route waypoints on the map before running the actual simulations. Go [here](https://github.com/AndreasKing-Goks/MAR-AST/tree/main/test_beds/map_and_route_plotter).

| Script Name | Description |
|----------|-------------|
| `plot_map_route.py` | Plot all the routes stored in `test_beds\map_route_plotter\data` along with map manually designed using `PolygonObstacle()` class.  |
| `plot_realmap_route.py` | Plot all the routes stored in `data\route` along with real-world map retrieved from [*Open Street Map*](https://www.openstreetmap.org/) |

### `ship_simu_test`
Consists of scripts for using the `env` class in `env_wrappers`. Go [here](https://github.com/AndreasKing-Goks/MAR-AST/tree/main/test_beds).

| Script Name | Description |
|----------|-------------|
| `test_multi_ship_map.py` | Test script for running `MultiShipEnv()` class on a real world map.  |
| `test_multi_ship.py` | Test script for running `MultiShipEnv()` class |
| `test_single_ship.py` | Test script for running `SingleShipEnv()` class |

---
