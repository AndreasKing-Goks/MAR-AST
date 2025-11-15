from pathlib import Path
import sys

## PATH HELPER (OBLIGATORY)
# project root = one levels up from this file
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

### IMPORT SIMULATOR ENVIRONMENTS
from run_script.setup import get_env_assets

## IMPORT FUNCTIONS
from utils.animate import MapAnimator, PolarAnimator, animate_side_by_side
from utils.plot_simulation import plot_ship_status, plot_ship_and_real_map

## IMPORT AST RELATED TOOLS
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from gymnasium.utils.env_checker import check_env

### IMPORT TOOLS
import argparse
import pandas as pd
import os
import time

### IMPORT UTILS
from utils.get_path import get_trained_model_and_log_path
from utils.logger import log_ast_training_config
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

    # Add arguments for AST-core
    parser.add_argument('--total_steps', type=int, default=100_000, metavar='TOTAL_STEPS',
                        help='AST: total steps of overall AST training [start_steps + train_steps] (default: 200_000)')
    parser.add_argument('--learning_rate', type=float, default=3e-4, metavar='LEARNING_RATE',
                        help='AST: learning rate for adam optimizer (default: 3e-4)')
    parser.add_argument('--buffer_size', type=int, default=1_000_000, metavar='REPLAY_BUFFER_SIZE',
                        help='AST: size of the replay buffer (default: 1_000_000)')
    parser.add_argument('--learning_starts', type=int, default=25_000, metavar='LEARNING_STARTS',
                        help='AST: how many steps of the model to collect transitions for before learning starts (default: 50_000)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='BATCH_SIZE',
                        help='AST: minibatch size for each gradient update (default: 256)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='SOFT_UPDATE_COEFFICIENT',
                        help='AST: the soft update coefficient [“Polyak update”, between 0 and 1] (default: 0.005)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='DISCOUNT_FACTOR',
                        help='AST: RL discount factor (default: 0.99)')
    parser.add_argument('--train_freq', type=int, default=1, metavar='TRAIN_FREQ',
                        help='AST: Update the model every train_freq steps. \
                            alternatively pass a tuple of frequency and unit like (5, "step") or (2, "episode") (default: 1)')
    parser.add_argument('--gradient_steps', type=int, default=1, metavar='GRADIENT_STEPS',
                        help='AST: How many gradient steps to do after each rollout (see train_freq). \
                            Set to -1 means to do as many gradient steps as steps done in the environment during the rollout (default: 1)')
    parser.add_argument('--ent_coef', type=str, default="auto", metavar='ENT_COEF',
                        help='AST: Entropy regularization coefficient. (Equivalent to inverse of reward scale in the original SAC paper.)\
                                Controlling exploration/exploitation trade-off. Set it to "auto" to learn it automatically \
                                (and "auto_0.1" for using 0.1 as initial value) (default: "auto")')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='TARGET_UPDATE_INTERVAL',
                        help='AST: update the target network every target_network_update_freq gradient steps (default: 1)')
    parser.add_argument('--target_entropy', type=str, default="auto", metavar='TARGET_ENTROPY',
                        help='AST: target entropy when learning ent_coef. Can be set to auto (default: "auto")')
    parser.add_argument('--stats_window_size', type=int, default=25, metavar='TARGET_UPDATE_INTERVAL',
                        help='AST: window size for the rollout logging, specifying the number of episodes to average \
                            the reported success rate, mean episode length, and mean reward over (default: 25)')
    parser.add_argument('--tensorboard_log', type=bool, default=True, metavar='TENSORBOARD_LOG',
                        help='AST: do tensorboard log. The log will be stored inside the training folder (default: True)')
    parser.add_argument('--verbose', type=int, default=1, metavar='VERBOSE',
                        help='AST: verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), \
                            2 for debug messages (default: 1)')
    parser.add_argument('--seed', type=int, default=None, metavar='SEED',
                        help='AST: seed for the pseudo random generators (default: None)')
    parser.add_argument('--device', type=str, default="cuda", metavar='DEVICE',
                        help='AST: device (cpu, cuda, …) on which the code should be run. \
                            Setting it to auto, the code will be run on the GPU if possible. (default: "cuda)')
    

    # Parse args
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

###################################### TRAIN THE MODEL #####################################

    # Path
    model_name  ="AST-train"
    model_path, log_path, tb_path = get_trained_model_and_log_path(root=ROOT, model_name=model_name)
    
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
    
    # Before training (so the config is at the top of the run)
    log_ast_training_config(args=args, txt_path=log_path, env=env, also_print=True)
    
    # Before model init
    if args.tensorboard_log:
        tb_dir = tb_path
    else:
        tb_dir = None
    
    ast_model = SAC("MultiInputPolicy",
                    env=Monitor(env),
                    learning_rate=args.learning_rate,
                    buffer_size=args.buffer_size,
                    learning_starts=args.learning_starts,
                    batch_size=args.batch_size,
                    tau=args.tau,
                    gamma=args.gamma,
                    train_freq=args.train_freq,
                    gradient_steps=args.gradient_steps,
                    ent_coef=args.ent_coef,
                    target_update_interval=args.target_update_interval,
                    target_entropy=args.target_entropy,
                    stats_window_size=args.stats_window_size,
                    tensorboard_log=tb_dir,
                    verbose=args.verbose,
                    seed=args.seed,
                    device=args.device)
    
    # Train the RL model. Record the time
    learn_kwargs = {}
    if tb_dir is not None:
        learn_kwargs["tb_log_name"] = "SAC_AST"
    start_time           = time.time()
    ast_model.learn(total_timesteps=args.total_steps, **learn_kwargs)
    elapsed_time         = time.time() - start_time
    raw_minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes       = divmod(raw_minutes, 60)
    train_time           = (hours, minutes, seconds)
    
    # Save the trained model
    ast_model.save(model_path)

################################## LOAD THE TRAINED MODEL ##################################

    # Remove the model to demonstrate saving and loading
    del ast_model
    
    # Load the trained model
    ast_model = SAC.load(model_path)
    
    ## Run the trained model
    obs, info = env.reset()
    while True:
        action, _states = ast_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
####################################### GET RESULTS ########################################

    # Print RL transition
    env.log_RL_transition_text(train_time=train_time,
                               txt_path=log_path,
                               also_print=True)
    env.log_RL_transition_json_csv(jsonl_path=log_path,
                                   csv_path=log_path,
                                   logger_name=model_name)

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