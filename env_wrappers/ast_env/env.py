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


class ASTEnv(gym.Env):
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
        self.wave_model = JONSWAPWaveModel(self.wave_model_config, seed=seed) if include_wave else None
        self.current_model = SurfaceCurrent(self.current_model_config, seed=seed) if include_current else None
        self.wind_model = NORSOKWindModel(self.wind_model_config, seed=seed) if include_wind else None
        
        # Ship drawing configuration
        self.ship_draw = args.ship_draw
        self.time_since_last_ship_drawing = args.time_since_last_ship_drawing
        
        # Environment termination flag
        self.ship_stop_status = [False] * len(self.assets)
        self.stop = False
        
        ### REINFORCEMENT LEARNING AGENT
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
        U_w_min, U_w_max                 = np.array([0.0, 32.9244444], dtype=np.float32)
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
        self.wave_range              = {"min": np.array([U_c_min, psi_c_min], dtype=np.float32), "max": np.array([U_w_max, psi_c_max], dtype=np.float32)}
        
        # Initialize action space
        self.init_action_space()
        
        # Initialize observation space
        self.init_observation_space()

        return
    
    def _normalize(self, x, min_val, max_val):
        """Normalize x from [min_val, max_val] to [-1, 1]."""
        return 2 * (x - min_val) / (max_val - min_val) - 1
    
    def init_action_space(self):
        ## Action Space (6)
        # Significant wave height
        Hs_min, Hs_max                   = [0.1, 15.0] 
        # Wave peak period
        Tp_min, Tp_max                   = [0.1, 23.7]
        # Mean wind speed
        U_w_bar_min, U_w_bar_max         = [0.0, 32.9244444] # in m/s. Knot [0, 64] 
        # Mean wind direction
        psi_ww_bar_min, psi_ww_bar_max   = [-np.pi, np.pi]
        # Mean current speed
        U_c_bar_min, U_c_bar_max         = [0.0, 5.0]
        # Mean current direction
        psi_c_bar_min, psi_c_bar_max     = [-np.pi, np.pi]
        
        self.action_space = Box(
            low  = np.array([Hs_min, Tp_min, U_w_bar_min, psi_ww_bar_min, U_c_bar_min, psi_c_bar_min], dtype=np.float32),
            high = np.array([Hs_max, Tp_max, U_w_bar_max, psi_ww_bar_max, U_c_bar_max, psi_c_bar_max], dtype=np.float32)
        )
        
    def init_observation_space(self):
        self.observation_space = gym.spaces.Dict(
            {
                "position"          : Box(-1.0, 1.0, shape=(3,)),
                "speed"             : Box(-1.0, 1.0, shape=(1,)),
                "cross_track_error" : Box(-1.0, 1.0, shape=(1,)),
                "wind"              : Box(-1.0, 1.0, shape=(2,)),
                "wave"              : Box(-1.0, 1.0, shape=(2,))
            }
        )
        
    def _get_obs(self):    
        # Get raw values
        position                = np.array([self.assets[0].ship_model.north, self.assets[0].ship_model.east, self.assets[0].ship_model.yaw_angle], dtype=np.float32)
        speed                   = np.array([self.assets[0].ship_model.speed], dtype=np.float32)
        cross_track_error       = np.array([self.assets[0].ship_model.auto_pilot.navigate.e_ct], dtype=np.float32)
        wind                    = np.array([self.wind_model.init_Ubar, self.wind_model.config.initial_wind_direction], dtype=np.float32)
        wave                    = np.array([self.current_model.config.initial_current_velocity, self.current_model.config.initial_current_direction], dtype=np.float32)
        
        position_norm           = self._normalize(position, self.position_range["min"], self.position_range["max"])
        speed_norm              = self._normalize(speed, self.speed_range["min"], self.speed_range["max"])
        cross_track_error_norm  = self._normalize(cross_track_error, self.cross_track_error_range["min"], self.cross_track_error_range["max"])
        wind_norm               = self._normalize(wind, self.wind_range["min"], self.wind_range["max"])
        wave_norm               = self._normalize(wave, self.wave_range["min"], self.wave_range["max"])

        observation         = {
            "position"          : position_norm,
            "speed"             : speed_norm,
            "cross_track_error" : cross_track_error_norm,
            "wind"              : wind_norm,
            "wave"              : wave_norm
        }
        
        return observation
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {

        }
    
    def step(self, action=None):
        ''' 
            The method is used for stepping up the simulator for the ship assets
            
            * Action unpcaked
            - Hs                : Significant wave height
            - Tp                : Wave peak period
            - U_w_bar           : Wind mean speed
            - psi_ww            : Wave and Wind mean direction
            - U_c_bar           : Current mean speed
            - psi_c             : Current mean direction
        '''
        ## Unpack the action
        Hs, Tp, U_w_bar, psi_ww, U_c_bar, psi_c = action
        
        ## GLOBAL ARGS FOR ALL SHIP ASSETS
        # Compile wave_args
        wave_args = self.wave_model.get_wave_force_params(Hs, Tp, psi_ww) if self.wave_model else None
        
        # Compile current_args
        current_args = self.current_model.get_current_vel_and_dir(U_c_bar, psi_c) if self.current_model else None
        
        # Compile wind_args
        wind_args = self.wind_model.get_wind_vel_and_dir(U_w_bar, psi_ww) if self.wind_model else None
        
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
        
        if np.all(self.ship_stop_status):
            self.stop = True
            
            observation = self._get_obs()
            reward      = 0.0
            terminated  = True
            truncated   = False
            info        = {}
            
            return observation, reward, terminated, truncated, info
        
        ## Apply ship drawing (set as optional function) after stepping
        if self.ship_draw:
            if self.time_since_last_ship_drawing > 30:
                for ship in self.assets:
                    ship.ship_model.ship_snap_shot()
                self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
            self.time_since_last_ship_drawing += self.args.time_step
            
        observation = self._get_obs()
        reward      = 0.0
        terminated  = False
        truncated   = False
        info        = {}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        ''' 
            Reset all of the ship environment inside the assets container.
        '''
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Reset the assets
        for i, asset in enumerate(self.assets):
            # Call upon the copied initial values
            init = asset.init_copy
            
            #  Reset the ship simulator
            asset.ship_model.reset()
            
            # Reset the asset info
            asset.info = init.info
        
        # Reset the stop status
        self.ship_stop_status = [False] * len(self.assets)
        self.stop = False
        
        # Reset the environment model
        if self.wave_model: self.wave_model.reset()
        if self.current_model: self.current_model.reset()
        if self.wind_model: self.wind_model.reset()
        
        # Reset the observation
        observation = self._get_obs()
        
        # Reset the info
        info = self._get_info()
        
        return observation, info