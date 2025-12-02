from pathlib import Path
import sys

## PATH HELPER (OBLIGATORY)
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from run_script.setup import get_env_assets

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
import numpy as np
import pandas as pd
import os
import time

### IMPORT UTILS
from utils.get_path import get_trained_model_path, get_saved_anim_path
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
    parser.add_argument('--nav_fail_time', type=int, default=600, metavar='NAV_FAIL_TIME',
                    help='ENV: Allowed recovery time in second from navigational failure warning condition (default: 600)')
    parser.add_argument('--traj_threshold_coeff', type=float, default=1.5, metavar='TRAJ_THRESHOLD_COEFF',
                    help='ENV: Coefficient to scale the maximum distance travelled based on the route segment length (default: 1.5)')
    parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                        help='ENV: record ship drawing for plotting and animation (default: True)')
    parser.add_argument('--time_since_last_ship_drawing', type=int, default=30, metavar='SHIP_DRAW_TIME',
                        help='ENV: time delay in second between ship drawing record (default: 30)')
    parser.add_argument('--map_gpkg_filename', type=str, default="Stangvik.gpkg", metavar='MAP_GPKG_FILENAME',
                        help='ENV: name of the .gpkg filename for the map (default: "Stangvik.gpkg")')
    parser.add_argument('--warm_up_time', type=int, default=2000, metavar='WARM_UP_TIME',
                        help='ENV: time needed in second before policy - action sampling takes place (default: 2000)')
    parser.add_argument('--action_sampling_period', type=int, default=1200, metavar='ACT_SAMPLING_PERIOD',
                        help='ENV: time period in second between policy - action sampling (default: 1200)')
    parser.add_argument('--max_sea_state', type=str, default="SS 5", metavar='MAX_SEA_STATE',
                        help='ENV: Maximum allowed sea state for environment model to condition the sea state table (default: "SS 5")')

    # Parse args
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

###################################### TRAIN THE MODEL #####################################
    # Path
    model_name  ="AST-train_2025-11-15_03-22-41_dd72"
    model_path, log_path = get_trained_model_path(root=ROOT, model_name=model_name)
    save_path = get_saved_anim_path(root=ROOT, model_name=model_name)
    
    # Get the args
    args = parse_cli_args()
    
    # Get the assets and AST Environment Wrapper
    env, assets, map_gdfs = get_env_assets(args=args, print_ship_status=False)
    
    # Set random route
    env.set_random_route_flag(flag=True)
    
    # Set for training flag
    env.set_for_training_flag(flag=False)
    
    # Load the trained model
    ast_model = SAC.load(model_path)

    from contracts.violation_collector import MultiRunContractLogger
    
    ## Run the trained model iteratively
    runs = 1000
    
    grounding_count                     = 0
    nav_failure_or_power_overload_count = 0
    exit_map_count                      = 0
    mission_finished_count              = 0
    action_invalid                      = 0

    multi_logger = MultiRunContractLogger()
    
    for run in range(runs):
        print(f"Run : {run + 1}")
        obs, info = env.reset()
        while True:
            action, _states = ast_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            stop_info = env.assets[0].ship_model.stop_info
            
            # Enable self.act_validity_list
            env.log_RL_transition_text()

            if terminated or truncated:
                if not np.all(env.act_validity_list):
                    action_invalid += 1
                    break

                else:
                    if stop_info['reaches_endpoint']:
                        mission_finished_count += 1
                    elif stop_info['grounding_failure']:
                        grounding_count += 1
                    elif stop_info['outside_horizon']:
                        exit_map_count += 1
                    elif stop_info['navigation_failure'] or stop_info['power_overload']:
                        nav_failure_or_power_overload_count += 1
                    else:
                        # optional: unknown / other category
                        pass
                break

        # After each run finishes:
        own_ship_results_df = pd.DataFrame.from_dict(env.assets[0].ship_model.simulation_results)

        # This calls your existing evaluate_contracts_over_dataframe(...)
        # and aggregates violations across runs in memory:
        multi_logger.run_once(
            df=own_ship_results_df,
            env=env,
            run_id=run + 1,   # or f"run_{run+1}"
        )

    total_count = grounding_count + nav_failure_or_power_overload_count + exit_map_count + mission_finished_count
####################################### GET RESULTS ########################################

def print_outcome_summary(total_count,
                          action_invalid,
                          grounding_count,
                          nav_failure_or_power_overload_count,
                          exit_map_count,
                          mission_finished_count):

    def pct(count):
        return (count / total_count * 100) if total_count > 0 else 0.0

    print("=" * 70)
    print(f"{'Outcome':35} {'Count':>10} {'Percent':>10}")
    print("-" * 70)
    print(f"{'Invalid Action Sampling':35} {action_invalid:10d} {pct(action_invalid):10.2f} %")
    print(f"{'Grounding':35} {grounding_count:10d} {pct(grounding_count):10.2f} %")
    print(f"{'Nav Failure or Power Overload':35} {nav_failure_or_power_overload_count:10d} {pct(nav_failure_or_power_overload_count):10.2f} %")
    print(f"{'Outside Horizon':35} {exit_map_count:10d} {pct(exit_map_count):10.2f} %")
    print(f"{'Mission Successful':35} {mission_finished_count:10d} {pct(mission_finished_count):10.2f} %")
    print("-" * 70)
    print(f"{'Total Count':35} {total_count:10d}")
    print("=" * 70)



    # ------------------------- 
    # After all runs: analysis
    # -------------------------
    all_violations_df = multi_logger.to_dataframe()
    print("All violations (head):")
    print(all_violations_df.head())

    print("\nTotal timesteps across all runs:", multi_logger.total_timesteps)

    print("\nPivot table (subsystem x contract_id):")
    pivot_table = multi_logger.summarize_by_contract()
    print(pivot_table)

    print("\nViolation rate by contract_id (% of timesteps):")
    print(multi_logger.violation_rate_by_contract())

    print("\nViolation rate by subsystem (% of timesteps):")
    print(multi_logger.violation_rate_by_subsystem())
    
print_outcome_summary(total_count, action_invalid, grounding_count, nav_failure_or_power_overload_count, exit_map_count, mission_finished_count)

