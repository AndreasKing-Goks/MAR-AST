from pathlib import Path
import sys

## PATH HELPER (OBLIGATORY)
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from test_beds.ast_test.setup import get_env_assets
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
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map

### IMPORT TOOLS
import argparse
from typing import List
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def parse_cli_args():
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
    parser.add_argument('--map_gpkg_filename', type=str, default='Stangvik.gpkg', metavar='MAP_FILENAME',
                        help="ENV: a string filename for the map file in gkpg format (default: 'Stangvik.gpkg')")

    # Add arguments for episodic run
    parser.add_argument('--n_episodes', type=int, default=1, metavar='N_EPISODES',
                        help='AST: number of simulation episode counts (default: 1)')

    # Parse args
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    
    # Get the args
    args = parse_cli_args()
    
    # Get the assets and AST Environment Wrapper
    env, assets, map_gdfs = get_env_assets(args=args)

    ### THIS IS WHERE THE EPISODE HAPPENS
    episode = 1
    while episode <= args.n_episodes:
        # Reset the environment at the beginning of episode
        env.reset()
        
        # Print message
        print("--- EPISODE " + str(episode) + " ---")
        
        ## THIS IS WHERE THE SIMULATION HAPPENS
        running_time = 0
        while running_time < assets[0].ship_model.int.sim_time and env.stop is False:
            env.step()
            
            # Update running time
            running_time = np.max([asset.ship_model.int.time for asset in assets])
        
        # Increment the episode
        episode += 1

    ################################## GET RESULTS ##################################

    # Build both animations (don’t show yet)
    repeat=False
    map_anim = MapAnimator(
        assets=assets,
        map_gdfs=map_gdfs,
        interval_ms=500,
        status_asset_index=0  # flags for own ship
    )
    map_anim.run(fps=120, show=False, repeat=repeat)

    polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
    polar_anim.run(fps=120, show=False, repeat=repeat)

    # Place windows next to each other, same height, centered
    animate_side_by_side(map_anim.fig, polar_anim.fig,
                        left_frac=0.68,  # how wide the map window is
                        height_frac=0.92,
                        gap_px=16,
                        show=True)

    ## Get the simulation results for all assets, and plot the asset simulation results
    result_dfs = []
    plot_env_load = [True, False, False] # Own ship, Target ship 1, Target ship 2
    for i, asset in enumerate(assets):
        result_df = pd.DataFrame().from_dict(env.assets[i].ship_model.simulation_results)
        result_dfs.append(result_df)
        
        # Plot 1: Status plot
        plot_ship_status(asset, result_df, plot_env_load=plot_env_load[i], show=False)

    # Plot 2: Ship and Map Plotting
    plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=True)