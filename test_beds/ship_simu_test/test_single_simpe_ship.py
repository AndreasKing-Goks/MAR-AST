from pathlib import Path
import sys

## PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def get_project_root():
    return str(ROOT)

def get_data_path(filename):
    return str(ROOT / "env_wrappers" / "simple_env" / "data" / filename)


### IMPORT SIMULATOR ENVIRONMENTS
from env_wrappers.simple_env.env import ShipAssets, SingleShipEnv

from simulator.ship_in_transit.sub_systems.ship_model import  ShipConfiguration, SimulationConfiguration, ShipModel
from simulator.ship_in_transit.sub_systems.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26, RudderConfiguration
from simulator.ship_in_transit.sub_systems.LOS_guidance import LosParameters
from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from simulator.ship_in_transit.sub_systems.controllers import ThrottleControllerGains, EngineThrottleFromSpeedSetPoint, HeadingControllerGains, HeadingBySampledRouteController
from simulator.ship_in_transit.sub_systems.wave_model import WaveModelConfiguration
from simulator.ship_in_transit.sub_systems.current_model import CurrentModelConfiguration
from simulator.ship_in_transit.sub_systems.wind_model import WindModelConfiguration

## IMPORT FUNCTIONS
from utils.animate import ShipTrajectoryAnimator, RLShipTrajectoryAnimator
from utils.center_plot import center_plot_window

### IMPORT TOOLS
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Argument Parser
parser = argparse.ArgumentParser(description='Ship in Transit Simulation')

## Add arguments for environments
parser.add_argument('--time_step', type=int, default=4, metavar='TIMESTEP',
                    help='ENV: time step size in second for ship transit simulator (default: 30)')
parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='ROA',
                    help='ENV: radius of acceptance for LOS algorithm (default: 300)')
parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='LD',
                    help='ENV: lookahead distance for LOS algorithm (default: 1000)')
parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                    help='ENV: record ship drawing for plotting and animation (default: True)')
parser.add_argument('--time_since_last_ship_drawing', default=30, metavar='SHIP_DRAW_TIME',
                    help='ENV: time delay in second between ship drawing record (default: 30)')

args = parser.parse_args()

# Engine configuration
main_engine_capacity = 3160e3 # 2160e3
diesel_gen_capacity = 610e3   # 510e3
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

### CONFIGURE THE SHIP SIMULATION MODELS
## Own ship
own_ship_config = SimulationConfiguration(
    initial_north_position_m=100,
    initial_east_position_m=100,
    initial_yaw_angle_rad=np.deg2rad(0.0),
    initial_forward_speed_m_per_s=4.0,
    initial_sideways_speed_m_per_s=0.0,
    initial_yaw_rate_rad_per_s=0.0,
    integration_step=args.time_step,
    simulation_time=10000,
)
# Set the throttle and autopilot controllers for the own ship
own_ship_throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=7, ki_ship_speed=0.13, kp_shaft_speed=0.05, ki_shaft_speed=0.005
)
own_ship_route_filename = 'own_ship_route.txt'
own_ship_route_name = get_data_path(own_ship_route_filename)
own_ship_heading_controller_gains = HeadingControllerGains(kp=2, kd=90, ki=0.001)
own_ship_los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
own_ship_desired_speed =8.0
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
    desired_speed=own_ship_desired_speed,
    engine_steps_per_time_step=10,
    initial_propeller_shaft_speed_rad_per_s=own_ship_initial_propeller_shaft_speed * np.pi /30,
    map_obj=map
)
# Wraps simulation objects based on the ship type using a dictionary
own_ship = ShipAssets(
    ship_model=own_ship,
)

# Package the assets for reinforcement learning agent
assets: List[ShipAssets] = [own_ship]

# Timer for drawing the ship
ship_draw = True
time_since_last_ship_drawing = 30

################################### ENV SPACE ###################################

# Initiate Multi-Ship Reinforcement Learning Environment Class Wrapper
env = SingleShipEnv(
    assets=assets,
    map=map,
    wave_model_config=wave_model_config,
    current_model_config=current_model_config,
    wind_model_config=wind_model_config,
    args=args)

# Test the simulation step up using policy's action sampling or direct action manipulation
test1 = True
# test1 = False

if test1:
    
    ## THIS IS WHERE THE LOOPING HAPPENS
    while own_ship.ship_model.int.time < own_ship.ship_model.int.sim_time and env.stop is False:
        env.step()

    # Get the simulation results for all assets
    own_ship_results_df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)

    # Plot 1: Overall process plot
    plot_1 = False
    plot_1 = True

    # Plot 2: Status plot
    plot_2 = False
    plot_2 = True

    # Create a No.2 3x4 grid for subplots
    if plot_2:
        fig_2, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        plt.figure(fig_2.number)  # Ensure it's the current figure
        axes = axes.flatten()  # Flatten the 2D array for easier indexing
    
        # Center plotting
        center_plot_window()

        # Plot 2.1:Speed
        own_ship_speed = np.sqrt(own_ship_results_df['forward speed [m/s]']**2 + own_ship_results_df['sideways speed [m/s]']**2)
        axes[0].plot(own_ship_results_df['time [s]'], own_ship_speed)
        axes[0].axhline(y=own_ship_desired_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
        axes[0].set_title('Own Ship Speed [m/s]')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Forward Speed (m/s)')
        axes[0].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[0].set_xlim(left=0)

        # Plot 2.2: Rudder Angle
        axes[1].plot(own_ship_results_df['time [s]'], own_ship_results_df['rudder angle [deg]'])
        axes[1].set_title('Test Ship Rudder angle [deg]')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Rudder angle [deg]')
        axes[1].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[1].set_xlim(left=0)
        axes[1].set_ylim(-31,31)

        # Plot 2.3: Cross Track error
        axes[2].plot(own_ship_results_df['time [s]'], own_ship_results_df['cross track error [m]'])
        axes[2].set_title('Own Ship Cross Track Error [m]')
        axes[2].axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Cross track error (m)')
        axes[2].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[2].set_xlim(left=0)
        axes[2].set_ylim(-501,501)

        # Plot 2.4: Propeller Shaft Speed
        axes[3].plot(own_ship_results_df['time [s]'], own_ship_results_df['propeller shaft speed [rpm]'])
        axes[3].set_title('Own Ship Propeller Shaft Speed [rpm]')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Propeller Shaft Speed (rpm)')
        axes[3].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[3].set_xlim(left=0)

        # Plot 2.5: Power vs Available Power
        if assets[0].ship_model.ship_machinery_model.operating_mode in ('PTO', 'MEC'):
            axes[4].plot(own_ship_results_df['time [s]'], own_ship_results_df['power me [kw]'], label="Power")
            axes[4].plot(own_ship_results_df['time [s]'], own_ship_results_df['available power me [kw]'], label="Available Power")
            axes[4].set_title("Own Ship's Power vs Available Mechanical Power [kw]")
            axes[4].set_xlabel('Time (s)')
            axes[4].set_ylabel('Power (kw)')
            axes[4].legend()
            axes[4].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[4].set_xlim(left=0)
        elif assets[0].ship_model.ship_machinery_model.operating_mode == 'PTI':
            axes[4].plot(own_ship_results_df['time [s]'], own_ship_results_df['power electrical [kw]'], label="Power")
            axes[4].plot(own_ship_results_df['time [s]'], own_ship_results_df['available power electrical [kw]'], label="Available Power")
            axes[4].set_title("Own Ship's Power vs Available Power Electrical [kw]")
            axes[4].set_xlabel('Time (s)')
            axes[4].set_ylabel('Power (kw)')
            axes[4].legend()
            axes[4].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[4].set_xlim(left=0)

        # Plot 2.6: Fuel Consumption
        axes[5].plot(own_ship_results_df['time [s]'], own_ship_results_df['fuel consumption [kg]'])
        axes[5].set_title('Own Ship Fuel Consumption [kg]')
        axes[5].set_xlabel('Time (s)')
        axes[5].set_ylabel('Fuel Consumption (kg)')
        axes[5].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[5].set_xlim(left=0)

        # Adjust layout for better spacing
        plt.tight_layout()

    if plot_1:    
        plt.figure(figsize=(10, 5.5))

        # Plot 1.1: Ship trajectory with sampled route
        # Test ship
        plt.plot(own_ship_results_df['east position [m]'].to_numpy(), own_ship_results_df['north position [m]'].to_numpy())
        plt.scatter(own_ship.ship_model.auto_pilot.navigate.east, own_ship.ship_model.auto_pilot.navigate.north, marker='x', color='blue')  # Waypoints
        plt.plot(own_ship.ship_model.auto_pilot.navigate.east, own_ship.ship_model.auto_pilot.navigate.north, linestyle='--', color='blue')  # Line
        for x, y in zip(own_ship.ship_model.ship_drawings[1], own_ship.ship_model.ship_drawings[0]):
            plt.plot(x, y, color='blue')
        map.plot_obstacle(plt.gca())  # get current Axes to pass into map function

        plt.xlim(0, 20000)
        plt.ylim(0, 10000)
        plt.title('Ship Trajectory')
        plt.xlabel('East position (m)')
        plt.ylabel('North position (m)')
        plt.gca().set_aspect('equal')
        plt.grid(color='0.8', linestyle='-', linewidth=0.5)

        # Adjust layout for better spacing
        plt.tight_layout()
    
    # Show Plot
    plt.show()