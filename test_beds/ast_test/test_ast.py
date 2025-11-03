from pathlib import Path
import sys

## PATH HELPER (OBLIGATORY)
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from test_beds.ast_test.setup import get_env_assets

## IMPORT FUNCTIONS
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map

## IMPORT AST RELATED TOOLS
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy, MultiInputPolicy
from gymnasium.wrappers import FlattenObservation, RescaleAction
from gymnasium.utils.env_checker import check_env

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
    parser.add_argument('--time_step', type=int, default=5, metavar='TIMESTEP',
                        help='ENV: time step size in second for ship transit simulator (default: 5)')
    parser.add_argument('--engine_step_count', type=int, default=10, metavar='ENGINE_STEP_COUNT',
                        help='ENV: engine integration step count in between simulation timestep (default: 10)')
    parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='ROA',
                        help='ENV: radius of acceptance in meter for LOS algorithm (default: 300)')
    parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='LD',
                        help='ENV: lookahead distance in meter for LOS algorithm (default: 1000)')
parser.add_argument('--nav_fail_time', type=int, default=300, metavar='NAV_FAIL_TIME',
                    help='ENV: Allowed recovery time in second from navigational failure warning condition (default: 300)')
    parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                        help='ENV: record ship drawing for plotting and animation (default: True)')
    parser.add_argument('--time_since_last_ship_drawing', default=30, metavar='SHIP_DRAW_TIME',
                        help='ENV: time delay in second between ship drawing record (default: 30)')

    # Add arguments for AST-core
    parser.add_argument('--n_episodes', type=int, default=1, metavar='N_EPISODES',
                        help='AST: number of simulation episode counts (default: 1)')
    parser.add_argument('--warm_up_time', default=1800, metavar='WARM_UP_TIME',
                        help='AST: time needed in second before policy - action sampling takes place (default: 1500)')
    parser.add_argument('--action_sampling_period', default=1800, metavar='ACT_SAMPLING_PERIOD',
                        help='AST: time period in second between policy - action sampling (default: 1000)')

    # Parse args
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    # Get the args
    args = parse_cli_args()
    
    # Get the assets and AST Environment Wrapper
    env, assets, map_gdfs = get_env_assets(args=args)
    
    # Get RL model
    ast_model = SAC("MultiInputPolicy",
                    env=env,
                    verbose=1,
                    device='cuda')
    # ast_model.learn(total_timesteps=1_000)
        
    # Get the trained agent to predict action
    ast_env = ast_model.get_env()
    obs = ast_env.reset()
    
    try:
        check_env(env)
        print("Environment passes all chekcs!")
    except Exception as e:
        print(f"Environment has issues: {e}")
    
    action, _ = ast_model.predict(obs, deterministic=True)
    
    observation, reward, terminated, truncated, info = env.step(action[0])
    
    print('obs          :', obs)
    print('observation  :', observation)
    print('reward       :', reward)
    print('terminated   :', terminated)
    print('truncated    :', truncated)

    # ### THIS IS WHERE THE EPISODE HAPPENS
    # episode = 1
    # while episode <= args.n_episodes:
    #     # Reset the environment at the beginning of episode
    #     env.reset()
        
    #     # Print message
    #     print("--- EPISODE " + str(episode) + " ---")
        
    #     ## THIS IS WHERE THE SIMULATION HAPPENS
    #     running_time = 0
    #     while running_time < assets[0].ship_model.int.sim_time and env.stop is False:
    #         env.step()
            
    #         # Update running time
    #         running_time = np.max([asset.ship_model.int.time for asset in assets])
        
    #     # Increment the episode
    #     episode += 1

    # ################################## GET RESULTS ##################################

    # # Build both animations (don’t show yet)
    # repeat=False
    # map_anim = MapAnimator(
    #     assets=assets,
    #     map_gdfs=map_gdfs,
    #     interval_ms=500,
    #     status_asset_index=0  # flags for own ship
    # )
    # map_anim.run(fps=120, show=False, repeat=repeat)

    # polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
    # polar_anim.run(fps=120, show=False, repeat=repeat)

    # # Place windows next to each other, same height, centered
    # animate_side_by_side(map_anim.fig, polar_anim.fig,
    #                     left_frac=0.68,  # how wide the map window is
    #                     height_frac=0.92,
    #                     gap_px=16,
    #                     show=True)

    # ## Get the simulation results for all assets, and plot the asset simulation results
    # result_dfs = []
    # plot_env_load = [True, False, False] # Own ship, Target ship 1, Target ship 2
    # for i, asset in enumerate(assets):
    #     result_df = pd.DataFrame().from_dict(env.assets[i].ship_model.simulation_results)
    #     result_dfs.append(result_df)
        
    #     # Plot 1: Status plot
    #     plot_ship_status(asset, result_df, plot_env_load=plot_env_load[i], show=False)

    # # Plot 2: Ship and Map Plotting
    # plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=True)
    
    