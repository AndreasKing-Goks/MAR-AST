# Waypoint Manager FMU (Python, FMI 2.0 Co-Simulation)

An FMI 2.0 Co-Simulation Functional Mock‑up Unit (FMU) that manages a user‑defined trajectory of waypoints and exposes the previous and next waypoint (with desired speed) based on a radius of acceptance. It supports waypoints supplied either in local NED meters or in geographic Lat/Lon degrees (auto‑converted at initialization). A final heading can be commanded once the last waypoint becomes active. A lightweight GeoJSON (LineString) of the full route is emitted for visualization.

This FMU is intended to be coupled with guidance / control FMUs (e.g. LOS guidance, speed controllers) but keeps its own minimal logic for waypoint switching and speed interpolation.

## Table of Contents
1. Key Features
2. Architecture & Data Flow
3. Dependencies
4. Building the FMU
5. Using the FMU (Parameters & Inputs & Outputs)
6. License

---

## 1. Key Features
- Dual coordinate input support: set `ned_coord = True` for NED meters, `False` for Lat/Lon degrees.
- Automatic Lat/Lon → local flat earth (NED) conversion (`llh2flat`) using reference latitude & longitude.
- Radius-of-acceptance switching (`r_accept`) selects the next waypoint when the vessel comes within range.
- Maintains and outputs both previous and next waypoint (position + desired speed).
- Up to 6 configurable waypoints; subset selected via `nu_of_wps`.
- Final heading activation: when last waypoint is active, `last_wp_active = True` and `final_heading_rad` is published.
- GeoJSON export of full trajectory (`traj_plan_for_visualization`) for plotting.
- Angle wrapping helpers (`wrap_to_pi`) keep headings within [-π, π].
- Speed interpolation utility (`calculate_desired_speed`) prepared for optional gradual speed transition (can be extended).


## 2. Architecture & Data Flow

### Core Methods (`waypoint_manager.py`)
- `wrap_to_pi(angle)`: Normalizes angles to [-π, π].
- `llh2flat(lat, lon, sog)`: Converts Lat/Lon to local north/east (flat earth) and passes through speed.
- `flat2llh(x_north, y_east)`: Reverse conversion for visualization / GeoJSON when starting from NED.
- `calculate_desired_speed(wp_leg_distance, ship_to_wp_distance, prev_speed, next_speed)`: Linear interpolation placeholder for additive speed profiling.
- `to_json()`: Builds GeoJSON `LineString` of full trajectory in Lat/Lon for external visualization.
- `exit_initialization_mode()`: Collects waypoint parameters, trims to `nu_of_wps`, converts to NED if needed, initializes previous/next waypoint outputs.
- `do_step(current_time, step_size)`: Performs waypoint switching if within `r_accept`; sets `last_wp_active` & `final_heading_rad` when final waypoint becomes current target.

### Data Flow
1. Parameters are set (either via FMU instantiation or master tool) before exiting initialization.
2. `exit_initialization_mode` builds `traj_plan` as a list of `(x_north, y_east, sog)` tuples.
3. At each co-simulation step, `do_step` checks distance to future waypoints, updating previous/next selections.
4. When the last waypoint is the active target, `last_wp_active` flips to `True` and final heading is exposed in radians.
5. External guidance FMU consumes previous/next waypoint + speed to compute desired course & speed.


## 3. Dependencies
Python ≥ 3.11 recommended (tested with 3.12). Minimal runtime requirement in `requirements.txt`:
- `numpy >= 1.26`

To build the FMU you also need `pythonfmu` installed in the environment (add to requirements if desired):
- `pythonfmu >= 0.6.7`

Install locally:
```powershell
pip install numpy>=1.26 pythonfmu>=0.6.7
```


## 4. Building the FMU
Use the `pythonfmu` CLI from the directory containing `waypoint_manager.py` & `requirements.txt`.
```powershell
python -m pythonfmu build -f waypoint_manager.py requirements.txt
```
The resulting `.fmu` will be placed in a `dist` (or similar) folder created by `pythonfmu`.


## 5. Using the FMU (Parameters, Inputs & Outputs)

### Coordinate Convention
For each waypoint i (01..06):
- If `ned_coord = False`: `wp_0i_x_north_lat` holds latitude (deg), `wp_0i_y_east_lon` holds longitude (deg).
- If `ned_coord = True`: they are already local N (m) and E (m) coordinates.

### Parameters
- `ned_coord = False` : Use Lat/Lon degrees (False) or NED meters (True) for waypoint parameter interpretation.
- `lat_reference = 0.0` : Reference latitude (deg) for flat-earth projection.
- `lon_reference = 0.0` : Reference longitude (deg) for flat-earth projection.
- `r_accept = 100.0` : Radius (m) for switching to next waypoint.
- `nu_of_wps = 2` : Number of active waypoints (1–6).
- `wp_01_x_north_lat`, `wp_01_y_east_lon`, `wp_01_sog` : Waypoint 1 position & desired speed (m/s).
- `wp_02_x_north_lat`, `wp_02_y_east_lon`, `wp_02_sog` : Waypoint 2.
- `wp_03_x_north_lat`, `wp_03_y_east_lon`, `wp_03_sog` : Waypoint 3.
- `wp_04_x_north_lat`, `wp_04_y_east_lon`, `wp_04_sog` : Waypoint 4.
- `wp_05_x_north_lat`, `wp_05_y_east_lon`, `wp_05_sog` : Waypoint 5.
- `wp_06_x_north_lat`, `wp_06_y_east_lon`, `wp_06_sog` : Waypoint 6.
- `final_heading_deg = 0.0` : Desired final heading (deg) published when last waypoint active.

### Inputs
- `x_north` : Current vessel north position (m) in local frame.
- `y_east`  : Current vessel east position (m) in local frame.

### Outputs
- `prev_wp_x_north`, `prev_wp_y_east`, `prev_wp_sog` : Previous waypoint (N, E, desired speed m/s).
- `next_wp_x_north`, `next_wp_y_east`, `next_wp_sog` : Next (target) waypoint (N, E, desired speed m/s).
- `last_wp_active` : Boolean flag set True when targeting final waypoint.
- `final_heading_rad` : Final heading converted to radians (wrapped [-π, π)).
- `traj_plan_for_visualization` : GeoJSON `LineString` of full path for plotting.

### Typical Usage Flow
1. Set reference lat/lon and choose coordinate mode (`ned_coord`).
2. Populate waypoint parameters and `nu_of_wps`.
3. Initialize FMU (master calls exit initialization mode).
4. At each simulation step feed current `x_north`, `y_east`.
5. Consume `next_wp_*` & `prev_wp_*` for guidance logic; monitor `last_wp_active` & `final_heading_rad`.


## 6. License
Currently unspecified. Recommendation: add a `LICENSE` file (MIT or Apache‑2.0) to enable open collaboration and reuse.

---
### Attribution and contact
Author: Melih Akdağ (melih.akdag@dnv.com)  
Date: November 2025



