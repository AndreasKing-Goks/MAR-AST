""" 
This modules provide classes for simple simulation running for one ship
"""
import numpy as np

from simulator.ship_in_transit.sub_systems.ship_model import ShipModel
from simulator.ship_in_transit.sub_systems.wave_model import JONSWAPWaveModel, WaveModelConfiguration
from simulator.ship_in_transit.sub_systems.current_model import SurfaceCurrent, CurrentModelConfiguration
from simulator.ship_in_transit.sub_systems.wind_model import NORSOKWindModel, WindModelConfiguration
from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from simulator.ship_in_transit.sub_systems.sbmpc import SBMPC


from simulator.ship_in_transit.utils import check_condition

from dataclasses import dataclass, field
from typing import Union, List

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


class MultiShipEnv:
    """
    This class is the main class for the Ship-Transit Simulator for multiple ships. It handles:
    
    To turn on collision avoidance on the ship under test:
    - set colav_mode=None         : No collision avoidance is implemented
    - set colav_mode='simple'     : Simple collision avoidance is implemented
    - set colav_mode='sbmpc'      : SBMPC collision avoidance is implemented
    """
    def __init__(self, 
                 assets:List[ShipAsset],
                 map: PolygonObstacle,
                 wave_model_config: WaveModelConfiguration,
                 current_model_config: CurrentModelConfiguration,
                 wind_model_config: WindModelConfiguration,
                 args):
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
        self.map = map 
        
        # Set configuration as an attribute
        self.wave_model_config = wave_model_config
        self.current_model_config = current_model_config
        self.wind_model_config = wind_model_config
        
        # Get the environment model based on the config
        self.wave_model = JONSWAPWaveModel(self.wave_model_config)
        self.current_model = SurfaceCurrent(self.current_model_config)
        self.wind_model = NORSOKWindModel(self.wind_model_config)  
        
        ## Fixed environment parameter
        # Wave
        self.Hs = 0.1
        self.Tp = 7.5
        self.psi_0 = np.deg2rad(45.0)
        
        # Current
        self.vel_mean = 1.0
        self.current_dir_mean = np.deg2rad(45.0)
        
        # Wind
        self.Ubar_mean = 1.0
        self.wind_dir_mean = np.deg2rad(-90)
        
        # Ship drawing configuration
        self.ship_draw = args.ship_draw
        self.time_since_last_ship_drawing = args.time_since_last_ship_drawing
        
        # Environment termination flag
        self.ship_stop_status = [False] * len(self.assets)
        self.stop = False
        
        return
    
    def step(self):
        ''' 
            The method is used for stepping up the simulator for the ship asset
        '''
        ## GLOBAL ARGS FOR ALL SHIP ASSETS
        # Compile wave_args
        wave_args = self.wave_model.get_wave_force_params(self.Hs, self.Tp, self.psi_0)
        
        # Compile current_args
        current_args = self.current_model.get_current_vel_and_dir(self.vel_mean, self.current_dir_mean)
        
        # Compile wind_args
        wind_args = self.wind_model.get_wind_vel_and_dir(self.Ubar_mean, self.wind_dir_mean)
        
        # Compile env_args
        env_args = (wave_args, current_args, wind_args)
        
        # Collect assets_info
        asset_infos = [asset.info for asset in self.assets]
        
        ## Step up all available digital assets
        for i, asset in enumerate(self.assets):
            # Step
            if asset.ship_model.stop is False: asset.ship_model.step(env_args, asset_infos)   # If all asset is not stopped, step up
            
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
            return
        
        ## Apply ship drawing (set as optional function) after stepping
        if self.ship_draw:
            if self.time_since_last_ship_drawing > 30:
                for ship in self.assets:
                    ship.ship_model.ship_snap_shot()
                self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
            self.time_since_last_ship_drawing += self.args.time_step
        
        return

    def reset(self):
        ''' 
            Reset all of the ship environment inside the assets container.
        '''
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
        self.wave_model.reset()
        self.current_model.reset()
        self.wind_model.reset()
        
        return
       