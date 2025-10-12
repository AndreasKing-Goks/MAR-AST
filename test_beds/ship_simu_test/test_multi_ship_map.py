from pathlib import Path
import sys
import geopandas as gpd

## PATH HELPER (OBLIGATORY)
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from env_wrappers.multiship_env.env import AssetInfo, ShipAsset, MultiShipEnv

from simulator.ship_in_transit.sub_systems.ship_model import  ShipConfiguration, SimulationConfiguration, ShipModel
from simulator.ship_in_transit.sub_systems.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26, RudderConfiguration
from simulator.ship_in_transit.sub_systems.LOS_guidance import LosParameters
from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from simulator.ship_in_transit.sub_systems.controllers import ThrottleControllerGains, HeadingControllerGains   
from simulator.ship_in_transit.sub_systems.wave_model import WaveModelConfiguration
from simulator.ship_in_transit.sub_systems.current_model import CurrentModelConfiguration
from simulator.ship_in_transit.sub_systems.wind_model import WindModelConfiguration

## IMPORT FUNCTIONS
from utils.get_path import get_ship_route_path, get_map_path
from utils.prepare_map import get_gdf_from_gpkg, get_polygon_from_gdf
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map

### IMPORT TOOLS
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


###############################################################################

# Argument Parser
parser = argparse.ArgumentParser(description='Ship in Transit Simulation')

## Add arguments for environments
parser.add_argument('--time_step', type=int, default=4, metavar='TIMESTEP',
                    help='ENV: time step size in second for ship transit simulator (default: 30)')
parser.add_argument('--engine_step_count', type=int, default=10, metavar='ENGINE_STEP_COUNT',
                    help='ENV: engine integration step count in between simulation timestep (default: 300)')
parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='ROA',
                    help='ENV: radius of acceptance for LOS algorithm (default: 300)')
parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='LD',
                    help='ENV: lookahead distance for LOS algorithm (default: 1000)')
parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                    help='ENV: record ship drawing for plotting and animation (default: True)')
parser.add_argument('--time_since_last_ship_drawing', default=30, metavar='SHIP_DRAW_TIME',
                    help='ENV: time delay in second between ship drawing record (default: 30)')

# Add arguments for episodic run
parser.add_argument('--n_episodes', type=int, default=1, metavar='N_EPISODES',
                    help='AST: number of simulation episode counts (default: 1)')

args = parser.parse_args()

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

map_data = get_polygon_from_gdf(land_gdf)   # list of exterior rings (E,N)
map = PolygonObstacle(map_data)              # <-- reuse your existing simulator map type

# Engine configuration
main_engine_capacity = 2160e3 #4160e3
diesel_gen_capacity = 510e3 #610e3
hybrid_shaft_gen_as_generator = 'GEN'
hybrid_shaft_gen_as_motor = 'MOTOR'
hybrid_shaft_gen_as_offline = 'OFF'

# Configure the simulation
ship_config = ShipConfiguration(
    coefficient_of_deadweight_to_displacement=0.7,
    bunkers=200000,
    ballast=200000,
    length_of_ship=80,
    width_of_ship=16,
    added_mass_coefficient_in_surge=0.4,
    added_mass_coefficient_in_sway=0.4,
    added_mass_coefficient_in_yaw=0.4,
    dead_weight_tonnage=3850000,
    mass_over_linear_friction_coefficient_in_surge=130,
    mass_over_linear_friction_coefficient_in_sway=18,
    mass_over_linear_friction_coefficient_in_yaw=90,
    nonlinear_friction_coefficient__in_surge=2400,
    nonlinear_friction_coefficient__in_sway=4000,
    nonlinear_friction_coefficient__in_yaw=400
)
wave_model_config = WaveModelConfiguration(
    minimum_wave_frequency=0.4,
    maximum_wave_frequency=2.5,
    wave_frequency_discrete_unit_count=50,
    minimum_spreading_angle=-np.pi,
    maximum_spreading_angle=np.pi,
    spreading_angle_discrete_unit_count=10,
    spreading_coefficient=1,
    rho=1025.0,
    timestep_size=args.time_step
)
current_model_config = CurrentModelConfiguration(
    initial_current_velocity=5,
    current_velocity_standard_deviation=0.05,
    current_velocity_decay_rate=0.0025,
    initial_current_direction=np.deg2rad(-45),
    current_direction_standard_deviation=0.05,
    current_direction_decay_rate=0.005,
    timestep_size=args.time_step
)
wind_model_config = WindModelConfiguration(
    initial_mean_wind_velocity=None,                    # Set to None to use a mean wind component
    mean_wind_velocity_decay_rate=0.001,
    mean_wind_velocity_standard_deviation=0.5,
    initial_wind_direction=np.deg2rad(45.0),
    wind_direction_decay_rate=0.001,
    wind_direction_standard_deviation=0.03,
    minimum_mean_wind_velocity=0.0,
    maximum_mean_wind_velocity=100.0,
    minimum_wind_gust_frequency=0.06,
    maximum_wind_gust_frequency=0.4,
    wind_gust_frequency_discrete_unit_count=100,
    clip_speed_nonnegative=True,
    kappa_parameter=0.0026,
    U10=2.5,
    wind_evaluation_height=5.0,
    timestep_size=args.time_step
)
pto_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=0,
    shaft_generator_state=hybrid_shaft_gen_as_generator,
    name_tag='PTO'
)
pto_mode = MachineryMode(params=pto_mode_params)

pti_mode_params = MachineryModeParams(
    main_engine_capacity=0,
    electrical_capacity=2*diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_motor,
    name_tag='PTI'
)
pti_mode = MachineryMode(params=pti_mode_params)

mec_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_offline,
    name_tag='MEC'
)
mec_mode = MachineryMode(params=mec_mode_params)
mso_modes = MachineryModes(
    [pto_mode, mec_mode, pti_mode]
)
fuel_spec_me = SpecificFuelConsumptionWartila6L26()
fuel_spec_dg = SpecificFuelConsumptionBaudouin6M26Dot3()
machinery_config = MachinerySystemConfiguration(
    machinery_modes=mso_modes,
    machinery_operating_mode=1,
    linear_friction_main_engine=68,
    linear_friction_hybrid_shaft_generator=57,
    gear_ratio_between_main_engine_and_propeller=0.6,
    gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
    propeller_inertia=6000,
    propeller_diameter=3.1,
    propeller_speed_to_torque_coefficient=7.5,
    propeller_speed_to_thrust_force_coefficient=1.7,
    hotel_load=200000,
    rated_speed_main_engine_rpm=1000,
    rudder_angle_to_sway_force_coefficient=50e3,
    rudder_angle_to_yaw_force_coefficient=500e3,
    max_rudder_angle_degrees=30,
    specific_fuel_consumption_coefficients_me=fuel_spec_me.fuel_consumption_coefficients(),
    specific_fuel_consumption_coefficients_dg=fuel_spec_dg.fuel_consumption_coefficients()
)

### CONFIGURE THE SHIP SIMULATION MODELS
## Own ship
own_ship_route_filename = 'own_ship_route.txt'
own_ship_route_name = get_ship_route_path(ROOT, own_ship_route_filename)

start_E, start_N = np.loadtxt(own_ship_route_name)[0]  # expecting two columns: east, north

own_ship_config = SimulationConfiguration(
    initial_north_position_m=start_E,
    initial_east_position_m=start_N,
    initial_yaw_angle_rad=np.deg2rad(-30.0),
    initial_forward_speed_m_per_s=4.0,
    initial_sideways_speed_m_per_s=0.0,
    initial_yaw_rate_rad_per_s=0.0,
    integration_step=args.time_step,
    simulation_time=10000,
)
# Set the throttle and autopilot controllers for the own ship
own_ship_throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=5, ki_ship_speed=0.025, kp_shaft_speed=0.025, ki_shaft_speed=0.0005 #kp_ship_speed=5, ki_ship_speed=0.13, kp_shaft_speed=0.04, ki_shaft_speed=0.001
)

own_ship_heading_controller_gains = HeadingControllerGains(kp=1.5, kd=70, ki=0.001)
own_ship_los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
own_ship_desired_speed = 8.0
own_ship_cross_track_error_tolerance = 750
own_ship_initial_propeller_shaft_speed = 420
own_ship = ShipModel(
    ship_config=ship_config,
    simulation_config=own_ship_config,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    machinery_config=machinery_config,                       
    throttle_controller_gain=own_ship_throttle_controller_gains,
    heading_controller_gain=own_ship_heading_controller_gains,
    los_parameters=own_ship_los_guidance_parameters,
    name_tag='Own ship',
    route_name=own_ship_route_name,
    engine_steps_per_time_step=args.engine_step_count,
    initial_propeller_shaft_speed_rad_per_s=own_ship_initial_propeller_shaft_speed * np.pi /30,
    desired_speed=own_ship_desired_speed,
    cross_track_error_tolerance=own_ship_cross_track_error_tolerance,
    map_obj=map,
    colav_mode='sbmpc'
)
own_ship_info = AssetInfo(
    # dynamic state (mutable)
    current_north       = own_ship.north,
    current_east        = own_ship.east,
    current_yaw_angle   = own_ship.yaw_angle,
    forward_speed       = own_ship.forward_speed,
    sideways_speed      = own_ship.sideways_speed,

    # static properties (constants)
    name_tag            = own_ship.name_tag,
    ship_length         = own_ship.l_ship,
    ship_width          = own_ship.w_ship
)
# Wraps simulation objects based on the ship type using a dictionary
own_ship_asset = ShipAsset(
    ship_model=own_ship,
    info=own_ship_info
)

## Target ship 1
tar_ship_route_filename1 = 'tar_ship_route_1.txt'
tar_ship_route_name1 = get_ship_route_path(ROOT, tar_ship_route_filename1)

start_E1, start_N1 = np.loadtxt(tar_ship_route_name1)[0]  # expecting two columns: east, north

tar_ship_config1 = SimulationConfiguration(
    initial_north_position_m=start_E1,
    initial_east_position_m=start_N1,
    initial_yaw_angle_rad=np.deg2rad(30.0),
    initial_forward_speed_m_per_s=4.0,
    initial_sideways_speed_m_per_s=0.0,
    initial_yaw_rate_rad_per_s=0.0,
    integration_step=args.time_step,
    simulation_time=10000,
)
# Set the throttle and autopilot controllers for the own ship
tar_ship_throttle_controller_gains1 = ThrottleControllerGains(
    kp_ship_speed=5, ki_ship_speed=0.025, kp_shaft_speed=0.025, ki_shaft_speed=0.0005
)

tar_ship_heading_controller_gains1 = HeadingControllerGains(kp=1.5, kd=70, ki=0.001)
tar_ship_los_guidance_parameters1 = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
tar_ship_desired_speed1 = 8.0
tar_ship_cross_track_error_tolerance1 = 750
tar_ship_initial_propeller_shaft_speed1 = 420
tar_ship1 = ShipModel(
    ship_config=ship_config,
    simulation_config=tar_ship_config1,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    machinery_config=machinery_config,                       
    throttle_controller_gain=tar_ship_throttle_controller_gains1,
    heading_controller_gain=tar_ship_heading_controller_gains1,
    los_parameters=tar_ship_los_guidance_parameters1,
    name_tag='Target ship 1',
    route_name=tar_ship_route_name1,
    engine_steps_per_time_step=args.engine_step_count,
    initial_propeller_shaft_speed_rad_per_s=tar_ship_initial_propeller_shaft_speed1 * np.pi /30,
    desired_speed=tar_ship_desired_speed1,
    cross_track_error_tolerance=tar_ship_cross_track_error_tolerance1,
    map_obj=map,
    colav_mode='sbmpc'
)
tar_ship_info1 = AssetInfo(
    # dynamic state (mutable)
    current_north       = tar_ship1.north,
    current_east        = tar_ship1.east,
    current_yaw_angle   = tar_ship1.yaw_angle,
    forward_speed       = tar_ship1.forward_speed,
    sideways_speed      = tar_ship1.sideways_speed,

    # static properties (constants)
    name_tag            = tar_ship1.name_tag,
    ship_length         = tar_ship1.l_ship,
    ship_width          = tar_ship1.w_ship
)
# Wraps simulation objects based on the ship type using a dictionary
tar_ship_asset1 = ShipAsset(
    ship_model=tar_ship1,
    info=tar_ship_info1
)

## Target ship 2
tar_ship_route_filename2 = 'tar_ship_route_2.txt'
tar_ship_route_name2 = get_ship_route_path(ROOT, tar_ship_route_filename2)

start_E2, start_N2 = np.loadtxt(tar_ship_route_name2)[0]  # expecting two columns: east, north

tar_ship_config2 = SimulationConfiguration(
    initial_north_position_m=start_E2,
    initial_east_position_m=start_N2,
    initial_yaw_angle_rad=np.deg2rad(-90.0),
    initial_forward_speed_m_per_s=4.0,
    initial_sideways_speed_m_per_s=0.0,
    initial_yaw_rate_rad_per_s=0.0,
    integration_step=args.time_step,
    simulation_time=10000,
)
# Set the throttle and autopilot controllers for the own ship
tar_ship_throttle_controller_gains2 = ThrottleControllerGains(
    kp_ship_speed=5, ki_ship_speed=0.025, kp_shaft_speed=0.025, ki_shaft_speed=0.0005
)

tar_ship_heading_controller_gains2 = HeadingControllerGains(kp=1.5, kd=70, ki=0.001)
tar_ship_los_guidance_parameters2 = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
tar_ship_desired_speed2 =8.0
tar_ship_cross_track_error_tolerance2 = 750
tar_ship_initial_propeller_shaft_speed2 = 420
tar_ship2 = ShipModel(
    ship_config=ship_config,
    simulation_config=tar_ship_config2,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    machinery_config=machinery_config,                       
    throttle_controller_gain=tar_ship_throttle_controller_gains2,
    heading_controller_gain=tar_ship_heading_controller_gains2,
    los_parameters=tar_ship_los_guidance_parameters2,
    name_tag='Target ship 2',
    route_name=tar_ship_route_name2,
    engine_steps_per_time_step=args.engine_step_count,
    initial_propeller_shaft_speed_rad_per_s=tar_ship_initial_propeller_shaft_speed2 * np.pi /30,
    desired_speed=tar_ship_desired_speed2,
    cross_track_error_tolerance=tar_ship_cross_track_error_tolerance2,
    map_obj=map,
    colav_mode='sbmpc'
)
tar_ship_info2 = AssetInfo(
    # dynamic state (mutable)
    current_north       = tar_ship2.north,
    current_east        = tar_ship2.east,
    current_yaw_angle   = tar_ship2.yaw_angle,
    forward_speed       = tar_ship2.forward_speed,
    sideways_speed      = tar_ship2.sideways_speed,

    # static properties (constants)
    name_tag            = tar_ship2.name_tag,
    ship_length         = tar_ship2.l_ship,
    ship_width          = tar_ship2.w_ship
)
# Wraps simulation objects based on the ship type using a dictionary
tar_ship_asset2 = ShipAsset(
    ship_model=tar_ship2,
    info=tar_ship_info2
)

# Package the assets for reinforcement learning agent
assets: List[ShipAsset] = [own_ship_asset, tar_ship_asset1, tar_ship_asset2]

# Timer for drawing the ship
ship_draw = True
time_since_last_ship_drawing = 30

################################### ENV SPACE ###################################

# Initiate Multi-Ship Reinforcement Learning Environment Class Wrapper
env = MultiShipEnv(
    assets=assets,
    map=map,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    args=args)


### THIS IS WHERE THE EPISODE HAPPENS
episode = 1
while episode <= args.n_episodes:
    # Reset the environment at the beginning of episode
    env.reset()
    
    # Print message
    print("--- EPISODE " + str(episode) + " ---")
    
    ## THIS IS WHERE THE SIMULATION HAPPENS
    running_time = np.max([asset.ship_model.int.time for asset in assets])
    while running_time < own_ship.int.sim_time and env.stop is False:
        env.step()
    
    # Increment the episode
    episode += 1

################################## GET RESULTS ##################################

## Get the simulation results for all assets, and plot the asset simulation results
result_dfs = []
plot_env_load = [True, False, False] # Own ship, Target ship 1, Target ship 2
for i, asset in enumerate(assets):
    result_df = pd.DataFrame().from_dict(env.assets[i].ship_model.simulation_results)
    result_dfs.append(result_df)
    
    # Plot 1: Status plot
    plot_ship_status(asset, result_df, plot_env_load=plot_env_load[i])

# Plot 1: Ship and Map Plotting
plot_ship_and_real_map(assets, result_dfs, land_gdf, ocean_gdf, water_gdf, coast_gdf, frame_gdf)

# Show Plot
plt.show()