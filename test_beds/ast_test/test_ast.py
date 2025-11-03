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

### IMPORT UTILS
from utils.get_path import get_saved_model_path
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
    parser.add_argument('--time_since_last_ship_drawing', type=int, default=30, metavar='SHIP_DRAW_TIME',
                        help='ENV: time delay in second between ship drawing record (default: 30)')
    parser.add_argument('--map_gpkg_filename', type=str, default="Stangvik.gpkg", metavar='MAP_GPKG_FILENAME',
                        help='ENV: name of the .gpkg filename for the map (default: "Stangvik.gpkg")')

    # Add arguments for AST-core
    parser.add_argument('--n_episodes', type=int, default=1, metavar='N_EPISODES',
                        help='AST: number of simulation episode counts (default: 1)')
    parser.add_argument('--warm_up_time', type=int, default=1800, metavar='WARM_UP_TIME',
                        help='AST: time needed in second before policy - action sampling takes place (default: 1500)')
    parser.add_argument('--action_sampling_period', type=int, default=1800, metavar='ACT_SAMPLING_PERIOD',
                        help='AST: time period in second between policy - action sampling (default: 1000)')

    # Parse args
    args = parser.parse_args()
    
    return args

def print_RL_transition(env):
    # Unpack observation
    north_list              = []
    east_list               = []
    heading_list            = []
    speed_list              = []
    cross_track_error_list  = []
    wind_speed_list         = []
    wind_dir_list           = []
    current_speed_list      = []
    current_dir_list        = []
    for obs in env.obs_list:
        # First denormalized obs
        obs = env._denormalize_observation(obs)
        
        north_list.append(obs["position"][0].item())
        east_list.append(obs["position"][1].item())
        heading_list.append(np.rad2deg(obs["position"][2]).item())
        speed_list.append(obs["speed"][0].item())
        cross_track_error_list.append(obs["cross_track_error"][0].item())
        wind_speed_list.append(obs["wind"][0].item())
        wind_dir_list.append(np.rad2deg(obs["wind"][1]).item())
        current_speed_list.append(obs["current"][0].item())
        current_dir_list.append(np.rad2deg(obs["current"][1]).item())
        
    # Unpack action
    Hs_list         = []
    Tp_list         = []
    U_w_bar_list    = []
    psi_ww_bar_list = []
    U_c_bar_list    = []
    psi_c_bar_list  = []
    for action in env.action_list:
        Hs_list.append(action[0].item())
        Tp_list.append(action[1].item())
        U_w_bar_list.append(action[2].item())
        psi_ww_bar_list.append(np.rad2deg(action[3]).item())
        U_c_bar_list.append(action[4].item())
        psi_c_bar_list.append(np.rad2deg(action[5]).item())
        
    # Do print
    print('#=========================== RL TRANSITION ==========================#')
    print('#---------------------------- Observation ---------------------------#')
    print('north             [m] :', north_list)
    print('east              [m] :', east_list)
    print('heading         [deg] :', heading_list)
    print('speed           [m/s] :', speed_list)
    print('cross track error [m] :', cross_track_error_list)
    print('wind speed      [m/s] :', wind_speed_list)
    print('wind dir        [deg] :', wind_dir_list)
    print('current speed   [m/s] :', current_speed_list)
    print('current dir     [deg] :', current_dir_list)
    print('#------------------------------ Action ------------------------------#')
    print('Hs                [m] :', Hs_list)
    print('Tp                [s] :', Tp_list)
    print('U_w_bar         [m/s] :', U_w_bar_list)
    print('psi_ww_bar      [deg] :', psi_ww_bar_list)
    print('U_c_bar         [m/s] :', U_c_bar_list)
    print('psi_c_Bar       [deg] :', psi_c_bar_list)
    print('#--------------------------------------------------------------------#')
    print('Terminated            :', env.terminated_list)
    print('#--------------------------------------------------------------------#')
    print('Truncated             :', env.truncated_list)
    print('#--------------------------------------------------------------------#')
    print('Reward                :', env.reward_list)
    print('#--------------------------------------------------------------------#')
    
    return

if __name__ == "__main__":

###################################### TRAIN THE MODEL #####################################

    # Get the args
    args = parse_cli_args()
    
    # Get the assets and AST Environment Wrapper
    env, assets, map_gdfs = get_env_assets(args=args)
    
    # Check the env if falid
    try:
        check_env(env)
        print("Environment passes all chekcs!")
    except Exception as e:
        print(f"Environment has issues: {e}")
        print("ABORT TRAINING")
        sys.exit(1)  # non-zero exit code stops the script
    
    # Set the Policy
    # Later
    
    # Set RL model
    ast_model = SAC("MultiInputPolicy",
                    env=env,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=100,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    action_noise=None,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    n_steps=1,
                    ent_coef="auto",
                    target_update_interval=1,
                    target_entropy="auto",
                    use_sde=False,
                    sde_sample_freq=-1,
                    use_sde_at_warmup=False,
                    stats_window_size=100,
                    tensorboard_log=None,
                    policy_kwargs=None,
                    verbose=1,
                    seed=None,
                    device='cuda')
    
    # Train the RL model
    ast_model.learn(total_timesteps=50)
    
    # Save the trained model
    saved_model_path = get_saved_model_path(root=ROOT, saved_model_filename="AST-trial_1")
    ast_model.save(saved_model_path)

################################## LOAD THE TRAINED MODEL ##################################

    # Remove the model to demonstrate saving and loading
    del ast_model
    
    # Load the trained model
    ast_model = SAC.load(saved_model_path)
    
    ## Run the trained model
    obs, info = env.reset()
    while True:
        action, _states = ast_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
####################################### GET RESULTS ########################################

    # Print RL transition
    print_RL_transition(env)

    ## Get the simulation results for all assets, and plot the asset simulation results
    own_ship_results_df = pd.DataFrame().from_dict(env.assets[0].ship_model.simulation_results)
    result_dfs = [own_ship_results_df]

    # Build both animations (don’t show yet)
    repeat=False
    map_anim = MapAnimator(
        assets=assets,
        map_gdfs=map_gdfs,
        interval_ms=500,
        status_asset_index=0  # flags for own ship
    )
    map_anim.run(fps=120, show=False, repeat=False)

    polar_anim = PolarAnimator(focus_asset=assets[0], interval_ms=500)
    polar_anim.run(fps=120, show=False, repeat=False)

    # Place windows next to each other, same height, centered
    animate_side_by_side(map_anim.fig, polar_anim.fig,
                        left_frac=0.68,  # how wide the map window is
                        height_frac=0.92,
                        gap_px=16,
                        show=True)

    # Plot 1: Trajectory
    plot_ship_status(env.assets[0], own_ship_results_df, plot_env_load=True, show=False)

    # Plot 2: Status plot
    plot_ship_and_real_map(assets, result_dfs, map_gdfs, show=True)