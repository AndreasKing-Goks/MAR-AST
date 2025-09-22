""" 
This modules provide classes for simple simulation running for one ship
"""
import numpy as np

from simulator.ship_in_transit.sub_systems.ship_model import ShipModel
from simulator.ship_in_transit.sub_systems.controllers import EngineThrottleFromSpeedSetPoint, HeadingBySampledRouteController
from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from simulator.ship_in_transit.sub_systems.sbmpc import SBMPC

from simulator.ship_in_transit.utils import check_condition

from dataclasses import dataclass, field
from typing import Union, List

import copy

@dataclass
class ShipAssets:
    ship_model: ShipModel
    throttle_controller: EngineThrottleFromSpeedSetPoint
    auto_pilot: HeadingBySampledRouteController
    desired_forward_speed: float
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
        
        # Set collision avoidance handle
        self.collav = args.collav_mode
        
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
    
    
    def reset(self):
        ''' 
            Reset all of the ship environment inside the assets container.
            
            Immediately call upon init_step() and init_get_intermediate_waypoint() method
            
            Return the initial state
            
            Note:
            When the action is not None, it means we immediately sample
            an intermediate waypoints for the obstacle ship to use
        '''
        # Reset the assets
        for i, ship in enumerate(self.assets):
            # Call upon the copied initial values
            init = ship.init_copy
            
            #  Reset the ship simulator
            ship.ship_model.reset()
            
            # Reset the ship throttle controller
            ship.throttle_controller.reset() 
            
            # Reset the autopilot controlller
            ship.auto_pilot.reset()
            
            # Reset parameters and lists
            ship.desired_forward_speed = init.desired_forward_speed
            ship.integrator_term = copy.deepcopy(init.integrator_term)
            ship.time_list = copy.deepcopy(init.time_list)
            
            # Reset the stop flag
            ship.stop_flag = False
        
        return
    
    
    def init_step(self):
        ''' 
            The initial step to place the ships at their initial states
            and to initiate the controller before running the simulator.
            
            Note:
            When the action is not None, it means we immediately sample
            an intermediate waypoints for the obstacle ship to use
        '''
        # For all assets
        for ship in self.assets:
            # Measure ship position and speed
            north_position = ship.ship_model.north
            east_position = ship.ship_model.east
            heading = ship.ship_model.yaw_angle
            forward_speed = ship.ship_model.forward_speed
        
            # Find appropriate rudder angle and engine throttle
            rudder_angle = ship.auto_pilot.rudder_angle_from_sampled_route(
                north_position=north_position,
                east_position=east_position,
                heading=heading
            )
        
            throttle = ship.throttle_controller.throttle(
                speed_set_point = ship.desired_forward_speed,
                measured_speed = forward_speed,
                measured_shaft_speed = forward_speed
            )
            
            # Store simulation data after init step
            ship.ship_model.store_simulation_data(throttle, 
                                                  rudder_angle,
                                                  ship.auto_pilot.get_cross_track_error(),
                                                  ship.auto_pilot.get_heading_error())
            
            # Step
            ship.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
            ship.ship_model.integrate_differentials()
            
            # Progress time variable to the next time step
            ship.ship_model.int.next_time()
                    
    
    def step(self):
        ''' 
            The method is used for stepping up the simulator for the ship assets
        '''          
        # Measure ship position and speed
        north_position = self.test.ship_model.north
        east_position = self.test.ship_model.east
        heading = self.test.ship_model.yaw_angle
        forward_speed = self.test.ship_model.forward_speed
        
        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.test.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading,
        )

        throttle = self.test.throttle_controller.throttle(
            speed_set_point = self.test.desired_forward_speed,
            measured_speed = forward_speed,
            measured_shaft_speed = forward_speed,
        )
        
        # Update and integrate differential equations for current time step
        self.test.ship_model.store_simulation_data(throttle, 
                                                   rudder_angle,
                                                   self.test.auto_pilot.get_cross_track_error(),
                                                   self.test.auto_pilot.get_heading_error())
        self.test.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        self.test.ship_model.integrate_differentials()
        
        self.test.integrator_term.append(self.test.auto_pilot.navigate.e_ct_int)
        self.test.time_list.append(self.test.ship_model.int.time)
        
        # Step up the simulator
        self.test.ship_model.int.next_time()
        
        # Apply ship drawing (set as optional function) after stepping
        if self.ship_draw:
            if self.time_since_last_ship_drawing > 30:
                self.test.ship_model.ship_snap_shot()
                self.obs.ship_model.ship_snap_shot()
                self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
            self.time_since_last_ship_drawing += self.test.ship_model.int.dt
        
        return