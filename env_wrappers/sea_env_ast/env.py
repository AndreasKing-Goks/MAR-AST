""" 
This module provides classes for AST-compliant environment wrapper
"""
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box

from simulator.ship_in_transit.sub_systems.ship_model import ShipModel
from simulator.ship_in_transit.sub_systems.wave_model import JONSWAPWaveModel, WaveModelConfiguration
from simulator.ship_in_transit.sub_systems.current_model import SurfaceCurrent, CurrentModelConfiguration
from simulator.ship_in_transit.sub_systems.wind_model import NORSOKWindModel, WindModelConfiguration
from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from simulator.ship_in_transit.sub_systems.env_load_prob_model import SeaStateMixture, logprior_mu_speed, logprior_mu_direction

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal

import copy

@dataclass
class AssetInfo:
    # dynamic state (mutable)
    current_north: float
    current_east: float
    current_yaw_angle: float
    forward_speed: float
    sideways_speed: float

    # static properties (constants)
    name_tag: str
    ship_length: float
    ship_width: float

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in ("name_tag", "ship_length", "ship_width"):
                raise AttributeError(f"{key} is constant and cannot be updated.")
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of {self.__class__.__name__}")

@dataclass
class ShipAsset:
    ship_model: ShipModel
    info: AssetInfo
    init_copy: 'ShipAsset' = field(default=None, repr=False, compare=False)


class SeaEnvAST(gym.Env):
    """
    This class is the main class for AST-compliant environment wrapper the Ship-Transit Simulator for multiple ships. It handles:
    
    To turn on collision avoidance on the ship under test:
    - set colav_mode=None         : No collision avoidance is implemented
    - set colav_mode='sbmpc'      : SBMPC collision avoidance is implemented
    """
    def __init__(self, 
                 assets:List[ShipAsset],
                 map: PolygonObstacle,
                 wave_model_config: WaveModelConfiguration,
                 current_model_config: CurrentModelConfiguration,
                 wind_model_config: WindModelConfiguration,
                 args,
                 include_wave=True,
                 include_current=True,
                 include_wind=True,
                 seed=None):
        '''
        Arguments:
        - assets    : List of all ship assets. 
                      First entry is always the ship under test
                      Second entry is/are the obstacle ship(s)
        - map       : Object map contains the location of land terrain
                      and its helper functions based on Shapely library
        - args      : Environmental arguments
        '''
        # Gymnasium-related attribute
        super().__init__()
        
        # Store args as attribute
        self.args = args
        
        ## Unpack assets
        self.assets = assets
        
        # Store initial values for each assets for reset function
        for _, asset in enumerate(self.assets):
            asset.init_copy=copy.deepcopy(asset)
        
        # Store the map class as attribute
        if map is not None:
            self.map = map[0]
            self.map_frame = map[1]
        
        # Set configuration as an attribute
        self.wave_model_config = wave_model_config
        self.current_model_config = current_model_config
        self.wind_model_config = wind_model_config
        
        # Get the environment model based on the config
        self.wave_model         = JONSWAPWaveModel(self.wave_model_config, seed=seed) if include_wave else None
        self.current_model      = SurfaceCurrent(self.current_model_config, seed=seed) if include_current else None
        self.wind_model         = NORSOKWindModel(self.wind_model_config, seed=seed) if include_wind else None
        self.sea_state_mixture  = SeaStateMixture()
        self.sea_state_mixture.condition_by_max_state(max_state_name="SS 5")
        
        # Previous sampled mean current speed, mean current direction, and mean wind-wave direction
        self.U_c_bar_prev       = self.current_model_config.initial_current_velocity
        self.psi_c_bar_prev     = self.current_model_config.initial_current_direction
        self.psi_ww_bar_prev    = self.wind_model_config.initial_wind_direction
        
        # Ship drawing configuration
        self.ship_draw = args.ship_draw
        self.time_since_last_ship_drawing = args.time_since_last_ship_drawing
        
        # Environment termination flag
        self.ship_stop_status = [False] * len(self.assets)
        self.terminated = False
        self.truncated  = False
        
        ### REINFORCEMENT LEARNING AGENT        
        ## Warm up environmental load states (Sea State 1, all direction to North)
        # Wave
        self.Hs_wu = 0.3 
        self.Tp_wu = 7.5 
        
        # Current
        self.U_c_bar_wu = 0.25
        self.psi_c_bar_wu = np.deg2rad(0.0)
        
        # Wind
        self.U_w_bar_wu = 1.5
        self.psi_ww_bar_wu = np.deg2rad(0.0)
        
        ## Observation space
        minx, miny, maxx, maxy           = self.map_frame.total_bounds
        # North ship position
        north_min, north_max             = np.array([miny, maxy], dtype=np.float32)
        # East ship position
        east_min, east_max               = np.array([minx, maxx], dtype=np.float32)
        # Ship heading (in NED)
        heading_min, heading_max         = np.array([-np.pi, np.pi], dtype=np.float32)
        # Ship speed
        speed_min, speed_max             = np.array([0.0, 20.0], dtype=np.float32)
        # LOS guidance cross track error
        e_ct_min, e_ct_max               = np.array([0.0, 3000.0], dtype=np.float32)
        # Wind speed
        U_w_min, U_w_max                 = np.array([0.0, 42.0], dtype=np.float32) # in m/s. Knot [0, ~80] 
        # Wind and Wave direction
        psi_ww_min, psi_ww_max           = np.array([-np.pi, np.pi], dtype=np.float32)
        # Current speed
        U_c_min, U_c_max                 = np.array([0.0, 5.0], dtype=np.float32)
        # Current direction
        psi_c_min, psi_c_max             = np.array([-np.pi, np.pi], dtype=np.float32)
        
        # Range for normalization
        self.position_range          = {"min": np.array([north_min, east_min, heading_min], dtype=np.float32), "max": np.array([north_max, east_max, heading_max], dtype=np.float32)}
        self.speed_range             = {"min": np.array([speed_min], dtype=np.float32), "max": np.array([speed_max], dtype=np.float32)}
        self.cross_track_error_range = {"min": np.array([e_ct_min], dtype=np.float32), "max": np.array([e_ct_max], dtype=np.float32)}
        self.wind_range              = {"min": np.array([U_w_min, psi_ww_min], dtype=np.float32), "max": np.array([U_w_max, psi_ww_max], dtype=np.float32)}
        self.current_range           = {"min": np.array([U_c_min, psi_c_min], dtype=np.float32), "max": np.array([U_c_max, psi_c_max], dtype=np.float32)}
        
        # Initialize action space
        self.init_action_space()
        
        # Initialize observation space
        self.init_observation_space()
        
        # RL transition containers
        self.obs_list           = []
        self.action_list        = []
        self.action_time_list   = []
        self.reward_list        = []
        self.terminated_list    = []
        self.truncated_list     = []
        self.info_list          = []

        return
    
    def _normalize(self, x, min_val, max_val):
        """Normalize x from [min_val, max_val] to [-1, 1]."""
        return 2 * (x - min_val) / (max_val - min_val) - 1
    
    def _denormalize(self, x_norm, min_val, max_val):
        """Denormalize x from [-1, 1] back to [min_val, max_val]."""
        return (x_norm + 1) * 0.5 * (max_val - min_val) + min_val

    def init_action_space(self):
        ## Action Space (6)
        # Significant wave height
        Hs_min, Hs_max                   = [0.1, 4.0] # [0.1, 15.0] 
        # Wave peak period
        Tp_min, Tp_max                   = [0.1, 15.5] # [0.1, 23.7]
        # Mean wind speed
        U_w_bar_min, U_w_bar_max         = [0.0, 12.87] # in m/s. Knot [0, ~27.0] # [0.0, 42.0] # in m/s. Knot [0, ~80]
        # Mean wind direction
        psi_ww_bar_min, psi_ww_bar_max   = [-np.pi, np.pi]
        # Mean current speed
        U_c_bar_min, U_c_bar_max         = [0.0, 1.0]
        # Mean current direction
        psi_c_bar_min, psi_c_bar_max     = [-np.pi, np.pi]
        
        # Range for normalization
        self.Hs_range           = np.array([Hs_min, Hs_max], dtype=np.float32)
        self.Tp_range           = np.array([Tp_min, Tp_max], dtype=np.float32)
        self.U_w_bar_range      = np.array([U_w_bar_min, U_w_bar_max], dtype=np.float32)
        self.psi_ww_bar_range   = np.array([psi_ww_bar_min, psi_ww_bar_max], dtype=np.float32)
        self.U_c_bar_range      = np.array([U_c_bar_min, U_c_bar_max], dtype=np.float32)
        self.psi_c_bar_range    = np.array([psi_c_bar_min, psi_c_bar_max], dtype=np.float32)
        
        self.action_space = Box(
            low  = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high = np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32)
        ) # In order -> [Hs, Tp, U_w_bar, psi_ww_bar, U_c_bar, psi_c_bar]
        
    def init_observation_space(self):
        self.observation_space = gym.spaces.Dict(
            {
                "position"          : Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
                "speed"             : Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "cross_track_error" : Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "wind"              : Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "current"           : Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            }
        )
    
    def _safe_clip(self, x, low=-1.0, high=1.0):
        # Prevent observation to go out of declared space's bound
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=high, neginf=low)
        return np.clip(x, low, high, dtype=np.float32)

    
    def _get_obs(self, normalized=True): 
        """
        Automatically normalized the observation.
        """   
        # Get raw values
        position                = np.array([self.assets[0].ship_model.north, self.assets[0].ship_model.east, self.assets[0].ship_model.yaw_angle], dtype=np.float32)
        speed                   = np.array([self.assets[0].ship_model.speed], dtype=np.float32)
        cross_track_error       = np.array([self.assets[0].ship_model.auto_pilot.navigate.e_ct], dtype=np.float32)
        wind                    = np.array([self.assets[0].ship_model.simulation_results['wind speed [m/s]'][-1], 
                                            self.assets[0].ship_model.simulation_results['wind dir [deg]'][-1]], dtype=np.float32)
        current                 = np.array([self.assets[0].ship_model.simulation_results['current speed [m/s]'][-1], 
                                            self.assets[0].ship_model.simulation_results['current dir [deg]'][-1]], dtype=np.float32)
        
        position_norm           = self._normalize(position, self.position_range["min"], self.position_range["max"])
        speed_norm              = self._normalize(speed, self.speed_range["min"], self.speed_range["max"])
        cross_track_error_norm  = self._normalize(cross_track_error, self.cross_track_error_range["min"], self.cross_track_error_range["max"])
        wind_norm               = self._normalize(wind, self.wind_range["min"], self.wind_range["max"])
        current_norm            = self._normalize(current, self.current_range["min"], self.current_range["max"])

        if normalized:
            # CLip the normalized observation within bound
            observation         = {
                "position"          : self._safe_clip(position_norm),
                "speed"             : self._safe_clip(speed_norm),
                "cross_track_error" : self._safe_clip(cross_track_error_norm),
                "wind"              : self._safe_clip(wind_norm),
                "current"           : self._safe_clip(current_norm)
            }
        else:
            observation         = {
                "position"          : position.astype(np.float32),
                "speed"             : speed.astype(np.float32),
                "cross_track_error" : cross_track_error.astype(np.float32),
                "wind"              : wind.astype(np.float32),
                "current"           : current.astype(np.float32)
            }
        
        return observation
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {

        }

    def _denormalize_action(self, action_norm):
        """
        Directly unpacks and denormalize the action from the RL agent
        """
        ## Unpack the action
        Hs_norm, Tp_norm, U_w_bar_norm, psi_ww_bar_norm, U_c_bar_norm, psi_c_bar_norm = action_norm # -> The action is nested
        
        ## Denormalize the action
        Hs = self._denormalize(Hs_norm, self.Hs_range[0], self.Hs_range[1])
        Tp = self._denormalize(Tp_norm, self.Tp_range[0], self.Tp_range[1])
        U_w_bar = self._denormalize(U_w_bar_norm, self.U_w_bar_range[0], self.U_w_bar_range[1])
        psi_ww_bar = self._denormalize(psi_ww_bar_norm, self.psi_ww_bar_range[0], self.psi_ww_bar_range[1])
        U_c_bar = self._denormalize(U_c_bar_norm, self.U_c_bar_range[0], self.U_c_bar_range[1])
        psi_c_bar = self._denormalize(psi_c_bar_norm, self.psi_c_bar_range[0], self.psi_c_bar_range[1])
        
        # Return action
        action = Hs, Tp, U_w_bar, psi_ww_bar, U_c_bar, psi_c_bar
        
        return action
    
    def _denormalize_observation(self, observation_norm):
        """
        Directly unpacks and denormalize the observation from the environment
        """    
        observation = {
            "position"              : self._denormalize(observation_norm["position"], self.position_range["min"], self.position_range["max"]),
            "speed"                 : self._denormalize(observation_norm["speed"], self.speed_range["min"], self.speed_range["max"]),
            "cross_track_error"     : self._denormalize(observation_norm["cross_track_error"], self.cross_track_error_range["min"], self.cross_track_error_range["max"]),
            "wind"                  : self._denormalize(observation_norm["wind"], self.wind_range["min"], self.wind_range["max"]),
            "current"               : self._denormalize(observation_norm["current"], self.current_range["min"], self.current_range["max"])
        }
        
        return observation    
    
    def _step(self, action=None):
        '''
            The method is used for stepping up the simulator for the ship assets
            
            * Action unpcaked
            - Hs                : Significant wave height
            - Tp                : Wave peak period
            - U_w_bar           : Wind mean speed
            - psi_ww_bar        : Wave and Wind mean direction
            - U_c_bar           : Current mean speed
            - psi_c_bar         : Current mean direction
        '''
        if action is not None:
            ## Unpack the action
            Hs, Tp, U_w_bar, psi_ww_bar, U_c_bar, psi_c_bar = action
            
            ## GLOBAL ARGS FOR ALL SHIP ASSETS
            # Compile wave_args
            wave_args = self.wave_model.get_wave_force_params(Hs, Tp, psi_ww_bar) if self.wave_model else None
            
            # Compile current_args
            current_args = self.current_model.get_current_vel_and_dir(U_c_bar, psi_c_bar) if self.current_model else None
            
            # Compile wind_args
            wind_args = self.wind_model.get_wind_vel_and_dir(U_w_bar, psi_ww_bar) if self.wind_model else None
            
            # Compile env_args
            env_args = (wave_args, current_args, wind_args)
        
        # No action means being in warm up phase       
        else:
            # Compile wave_args
            wave_args = self.wave_model.get_wave_force_params(self.Hs_wu, self.Tp_wu, self.psi_ww_bar_wu) if self.wave_model else None
            
            # Compile current_args
            current_args = self.current_model.get_current_vel_and_dir(self.U_c_bar_wu, self.psi_c_bar_wu) if self.current_model else None
            
            # Compile wind_args
            wind_args = self.wind_model.get_wind_vel_and_dir(self.U_w_bar_wu, self.psi_ww_bar_wu) if self.wind_model else None
            
            # Compile env_args
            env_args = (wave_args, current_args, wind_args)
            
        # Collect assets_info
        asset_infos = [asset.info for asset in self.assets]
        
        ## Step up all available digital assets
        for i, asset in enumerate(self.assets):
            # Step
            if asset.ship_model.stop is False: asset.ship_model.step(env_args=env_args, 
                                                                     asset_infos=asset_infos)   # If all asset is not stopped, step up
            
            # Update asset.info
            asset.info.update(current_north     = asset.ship_model.north,
                              current_east      = asset.ship_model.east,
                              current_yaw_angle = asset.ship_model.yaw_angle,
                              forward_speed     = asset.ship_model.forward_speed,
                              sideways_speed    = asset.ship_model.sideways_speed)
            
            # Update stop list
            self.ship_stop_status[i] = asset.ship_model.stop
        
        ## Apply ship drawing (set as optional function) after stepping
        if self.ship_draw:
            if self.time_since_last_ship_drawing > 30:
                for ship in self.assets:
                    ship.ship_model.ship_snap_shot()
                self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
            self.time_since_last_ship_drawing += self.args.time_step
        
        return
    
    def step(self, action_norm):
        ''' 
            The method is used to step up the Reinforcement Learning step.
        '''
        # Record the when tha action is sampled
        self.action_time_list.append(self.assets[0].ship_model.int.time)
        
        # Denormalize action
        action = self._denormalize_action(action_norm)
        
        # Unpack some of the action for environmental load memory
        _, _, _, psi_ww_bar, U_c_bar, psi_c_bar = action
        
        #------------------------------ Step the simulator ------------------------------#
        running_time = 0
        # Run the simulator within the action sampling period or until the own ship stopped.
        while running_time < self.args.action_sampling_period:
            self._step(action)
            
            # Update running time using simulator time step
            running_time += self.assets[0].ship_model.int.dt 
            
            # Check if all the ship assets has stopped
            if np.all(self.ship_stop_status):
                # Set the environment model termination flag as True if all the ship assets are stop
                self.terminated = True
                break
            
            # Check if the simulator still within the maximum simulation time
            if self.assets[0].ship_model.int.time > self.assets[0].ship_model.simulation_config.simulation_time:
                # Set the environment model truncated flag as True if all the ship assets not stoping within time limit
                self.truncated  = True
                break
        #--------------------------------------------------------------------------------#
        
        # Get the RL stepping outputs
        observation = self._get_obs()
        reward      = self.reward_function(action)
        terminated  = self.terminated
        truncated   = self.truncated
        info        = {}
        
        # Append the RL transition containers
        self.obs_list.append(observation)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.terminated_list.append(terminated)
        self.truncated_list.append(truncated)
        self.info_list.append(info)
        
        # Update the environmental load memory
        self.U_c_bar_prev       = U_c_bar
        self.psi_c_bar_prev     = psi_c_bar
        self.psi_ww_bar_prev    = psi_ww_bar
        
        return observation, reward, terminated, truncated, info
    
    def reward_function(self, action, logp_floor=-30.0):
        """
        For this reward function, we only take into account the own_ship
        """
        ## Unpack action
        [Hs, Tp, U_w_bar, psi_ww_bar, U_c_bar, psi_c_bar] = action
        
        ## Base reward -> Encourage further exploration [Can be a hyper parameter as well]
        reward = len(self.action_list) * 5.0
        
        ## Get the termination info of the own ship
        collision           = self.assets[0].ship_model.stop_info['collision']
        grounding_failure   = self.assets[0].ship_model.stop_info['grounding_failure']
        navigation_failure  = self.assets[0].ship_model.stop_info['navigation_failure']
        reaches_endpoint    = self.assets[0].ship_model.stop_info['reaches_endpoint']
        outside_horizon     = self.assets[0].ship_model.stop_info['outside_horizon']
        power_overload      = self.assets[0].ship_model.stop_info['power_overload']
        
        ## Get reward from the environmental load log probability
        # Sea state marginal log likelihood (clip to floor if we encounter log prob of negative infinity)
        sea_state_ll            = max(self.sea_state_mixture.logpdf_marginal(Hs, U_w_bar, Tp), logp_floor)
        
        # Current speed direction (clip to floor if we encounter log prob of negative infinity)
        current_speed_ll        = max(logprior_mu_speed(U_c_bar, range=(self.current_range["min"][0], self.current_range["max"][0]), center=self.U_c_bar_prev),
                                  logp_floor)
        
        # Current speed direction
        current_direction_ll    = logprior_mu_direction(psi_c_bar, clim_mean_dir=self.psi_c_bar_prev)
        
        # Wind speed direction
        wind_direction_ll       = logprior_mu_direction(psi_ww_bar, clim_mean_dir=self.psi_ww_bar_prev)
        
        # Sum all the log likelihood to get the reward_signal
        reward_env_ll = sea_state_ll + current_speed_ll + current_direction_ll + wind_direction_ll 
        
        # Add to the base reward
        reward += reward_env_ll
        
        ## Get reward from termination status
        if collision or navigation_failure or power_overload:
            reward += 10.0      # If failure gives positive reward
        elif grounding_failure:
            reward += 15.0      # We value grounding failure more
        elif outside_horizon:
            reward += 0.0       # Not very insightful
        elif reaches_endpoint:
            reward += -10.0     # We discourage the agent to let the ship finishes its mission.
        
        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        ''' 
            Reset all of the ship environment inside the assets container.
        '''
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Deterministically seed sub-models (use their own rng if they have one)
        if self.wave_model is not None:
            self.wave_model.reset(seed=int(self.np_random.integers(0, 2**31 - 1)))
        if self.current_model is not None:
            self.current_model.reset(seed=int(self.np_random.integers(0, 2**31 - 1)))
        if self.wind_model is not None:
            self.wind_model.reset(seed=int(self.np_random.integers(0, 2**31 - 1)))

        # Reset ships; pass seeds if supported
        for asset in self.assets:
            if hasattr(asset.ship_model, "reset"):
                try:
                    asset.ship_model.reset(seed=int(self.np_random.integers(0, 2**31 - 1)))
                except TypeError:
                    asset.ship_model.reset()

            # restore info to initial values (deep-copied)
            init = asset.init_copy
            asset.info = copy.deepcopy(init.info)
        
        # Reset the stop status
        self.ship_stop_status = [False] * len(self.assets)
        self.terminated = False
        self.truncated  = False
        
        #--------------------------------- Warm Up Phase --------------------------------#
        # Do step without implementing action for the warm up phase
        # BEWARE: 
        # MAKE SURE THAT DURING THE WARM UP PHASE, 
        # SIMULATOR SHOULD NOT BE TERMINATED/TRUNCATED
        running_time = 0
        while running_time < self.args.warm_up_time:
            # Simulator integration using a very gentle environment load
            self._step()
            
            # Update running time using simulator time step
            running_time += self.assets[0].ship_model.int.dt 
            
            # Check if all the ship assets has stopped
            if np.all(self.ship_stop_status):
                # Set the environment model termination flag as True if all the ship assets are stop
                self.terminated = True
                break
            
            # Check if the simulator still within the maximum simulation time
            if self.assets[0].ship_model.int.time > self.assets[0].ship_model.simulation_config.simulation_time:
                # Set the environment model truncated flag as True if all the ship assets not stoping within time limit
                self.truncated  = True
                break
        #--------------------------------------------------------------------------------#
        
        # Reset the environmental load memory. First memory is using the warm up environmental load's parameters
        self.U_c_bar_prev       = self.U_c_bar_wu
        self.psi_c_bar_prev     = self.psi_c_bar_wu
        self.psi_ww_bar_prev    = self.psi_ww_bar_wu
        
        # Reset the observation
        observation = self._get_obs()
        
        # Reset the info
        info = self._get_info()
        
        # Reset the RL transition containers
        self.obs_list           = [observation] # Immediately store the first observation from the reset
        self.action_list        = []
        self.action_time_list   = []
        self.reward_list        = []
        self.terminated_list    = []
        self.truncated_list     = []
        
        return observation, info
    
    def print_RL_transition(self):
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
        for obs in self.obs_list:
            # First denormalized obs
            obs = self._denormalize_observation(obs)
            
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
        Hs_list           = []
        Tp_list           = []
        U_w_bar_list      = []
        psi_ww_bar_list   = []
        U_c_bar_list      = []
        psi_c_bar_list    = []
        act_validity_list = []
        sea_state_list    = []
        for action in self.action_list:
            Hs      = action[0].item()
            Tp      = action[1].item()
            U_w_bar = action[3].item()
            act_validity = self.sea_state_mixture.action_validity(Hs, Tp, U_w_bar)
            if act_validity:
                idx       = self.sea_state_mixture.matching_states(Hs, Tp, U_w_bar)[0]
                sea_state = self.sea_state_mixture.states[idx]["name"]
            else:
                sea_state = None
            
            Hs_list.append(Hs)
            Tp_list.append(Tp)
            U_w_bar_list.append(U_w_bar)
            psi_ww_bar_list.append(np.rad2deg(action[3]).item())
            U_c_bar_list.append(action[4].item())
            psi_c_bar_list.append(np.rad2deg(action[5]).item())
            act_validity_list.append(act_validity)
            sea_state_list.append(sea_state)
            
        # Do print
        print('#===================================== RL TRANSITION ====================================#')
        print('#-------------------------------------- Observation -------------------------------------#')
        print('north                [m] :', north_list)
        print('east                 [m] :', east_list)
        print('heading            [deg] :', heading_list)
        print('speed              [m/s] :', speed_list)
        print('cross track error    [m] :', cross_track_error_list)
        print('wind speed         [m/s] :', wind_speed_list)
        print('wind dir           [deg] :', wind_dir_list)
        print('current speed      [m/s] :', current_speed_list)
        print('current dir        [deg] :', current_dir_list)
        print('#---------------------------------------- Action ----------------------------------------#')
        print('sampling timestamp   [s] :', self.action_time_list)
        print('Hs                   [m] :', Hs_list)
        print('Tp                   [s] :', Tp_list)
        print('U_w_bar            [m/s] :', U_w_bar_list)
        print('psi_ww_bar         [deg] :', psi_ww_bar_list)
        print('U_c_bar            [m/s] :', U_c_bar_list)
        print('psi_c_bar          [deg] :', psi_c_bar_list)
        print('action validity          :', act_validity_list)
        print('sea state                :', sea_state_list)
        print('#----------------------------------------------------------------------------------------#')
        print('Terminated               :', self.terminated_list)
        print('#----------------------------------------------------------------------------------------#')
        print('Truncated                :', self.truncated_list)
        print('#----------------------------------------------------------------------------------------#')
        print('Reward                   :', self.reward_list)
        print('#----------------------------------------------------------------------------------------#')
        
        return