""" 
This modules provide classes for simple simulation running for one ship
"""
import numpy as np

from simulator.ship_in_transit.sub_systems.ship_model import ShipModel
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
        
        # Step up all available digital assets
        self.own_ship.ship_model.step()
        
        # Apply ship drawing (set as optional function) after stepping
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
       