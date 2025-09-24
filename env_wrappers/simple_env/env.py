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
class ShipAssets:
    ship_model: ShipModel
    integrator_term: List[float]
    time_list: List[float]
    type_tag: str
    stop_flag: bool
    init_copy: 'ShipAssets' = field(default=None, repr=False, compare=False)


class SingleShipEnv:
    """
    This class is the main class for the Ship-Transit Simulator. It handles:
    
    To turn on collision avoidance on the ship under test:
    - set collav=None         : No collision avoidance is implemented
    - set collav='simple'     : Simple collision avoidance is implemented
    - set collav='sbmpc'      : SBMPC collision avoidance is implemented
    """
    def __init__(self, 
                 assets:List[ShipAssets],
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
        self.own_ship = self.assets[0]
        
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
        self.Hs = 0.3
        self.Tp = 7.5
        self.psi_0 = np.deg2rad(45)
        
        # Current
        self.vel_mean = 0.0
        self.current_dir_mean = 0.0
        
        # Wind
        self.Ubar_mean = 0.0
        self.wind_dir_mean = 0.0
        
        # Ship drawing configuration
        self.ship_draw = args.ship_draw
        self.time_since_last_ship_drawing = args.time_since_last_ship_drawing

        # Scenario-Based Model Predictive Controller
        self.sbmpc = SBMPC(tf=1000, dt=20)
        
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
        
        ## Step up all available digital assets
        self.own_ship.ship_model.step(env_args)
        
        ## Apply ship drawing (set as optional function) after stepping
        if self.ship_draw:
            if self.time_since_last_ship_drawing > 30:
                self.own_ship.ship_model.ship_snap_shot()
                self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
            self.time_since_last_ship_drawing += self.own_ship.ship_model.int.dt
        
        return

    def reset(self):
        ''' 
            Reset all of the ship environment inside the assets container.
        '''
        # Reset the assets
        for i, ship in enumerate(self.assets):
            # Call upon the copied initial values
            init = ship.init_copy
            
            #  Reset the ship simulator
            ship.ship_model.reset()
            
            # Reset parameters and lists
            ship.integrator_term = copy.deepcopy(init.integrator_term)
            ship.time_list = copy.deepcopy(init.time_list)
            
            # Reset the stop flag
            ship.stop_flag = False
        
        return
       