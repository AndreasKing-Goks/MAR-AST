""" 
Waypoint Manager Python FMU implementation.
This FMU manages a list of waypoints and provides the previous and next waypoints.
It supports waypoints defined in either NED coordinates (meters) or Lat/Lon (degrees).
It also includes a switching mechanism based on a radius of acceptance.

Authors: Melih Akdağ
Date: November 2025
"""

from pythonfmu import Fmi2Causality, Fmi2Slave, Fmi2Variability, Real, Integer, Boolean, String
import numpy as np
import math
from typing import Tuple
import json


class WaypointManager(Fmi2Slave):

    author = "Melih Akdağ"
    description = "Waypoint Manager Python FMU implementation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # PARAMETERS
        self.ned_coord = False      # Boolean parameter for conversion. Set it False if the waypoints are supplied in Lat/Lon in degrees and True if they are in NED in meters.
        self.lat_reference = 0.0    # Reference latitude position in degrees. Used for translating lat/lon to NED in meters.
        self.lon_reference = 0.0    # Reference longitude position in degrees. Used for translating lat/lon to NED in meters.
        
        self.r_accept = 100.0       # Radius of acceptance in meters. Used for switching mechanism.

        self.nu_of_wps = 2          # Number of waypoints defined by the user

        self.wp_01_y_east_lon = 0.0     # Waypoint 1, east in meters or longitude in degrees based on coordinate system
        self.wp_01_x_north_lat = 0.0    # Waypoint 1, north in meters or latitude in degrees based on coordinate system
        self.wp_01_sog = 0.0            # Desired speed at waypoint 1 in m/s

        self.wp_02_y_east_lon = 0.0     # Waypoint 2, east in meters or longitude in degrees based on coordinate system
        self.wp_02_x_north_lat = 0.0    # Waypoint 2, north in meters or latitude in degrees based on coordinate system
        self.wp_02_sog = 0.0            # Desired speed at waypoint 2 in m/s

        self.wp_03_y_east_lon = 0.0     # Waypoint 3, east in meters or longitude in degrees based on coordinate system
        self.wp_03_x_north_lat = 0.0    # Waypoint 3, north in meters or latitude in degrees based on coordinate system
        self.wp_03_sog = 0.0            # Desired speed at waypoint 3 in m/s

        self.wp_04_y_east_lon = 0.0     # Waypoint 4, east in meters or longitude in degrees based on coordinate system
        self.wp_04_x_north_lat = 0.0    # Waypoint 4, north in meters or latitude in degrees based on coordinate system
        self.wp_04_sog = 0.0            # Desired speed at waypoint 4 in m/s
    
        self.wp_05_y_east_lon = 0.0     # Waypoint 5, east in meters or longitude in degrees based on coordinate system
        self.wp_05_x_north_lat = 0.0    # Waypoint 5, north in meters or latitude in degrees based on coordinate system
        self.wp_05_sog = 0.0            # Desired speed at waypoint 5 in m/s

        self.wp_06_y_east_lon = 0.0     # Waypoint 6, east in meters or longitude in degrees based on coordinate system
        self.wp_06_x_north_lat = 0.0    # Waypoint 6, north in meters or latitude in degrees based on coordinate system
        self.wp_06_sog = 0.0            # Desired speed at waypoint 6 in m/s

        self.final_heading_deg = 0.0    # Final heading angle in degrees (0-360) when reached the goal

        # INPUT VARIABLES
        self.x_north = 0.0              # Current position, north in meters
        self.y_east = 0.0               # Current position, east in meters
        
        # OUTPUT VARIABLES
        self.prev_wp_x_north = 0.0      # Previous waypoint, north in meters
        self.prev_wp_y_east = 0.0       # Previous waypoint, east in meters
        self.prev_wp_sog = 0.0          # Previous waypoint, speed over ground in m/s

        self.next_wp_x_north = 0.0      # Next waypoint, north in meters
        self.next_wp_y_east = 0.0       # Next waypoint, east in meters
        self.next_wp_sog = 0.0          # Next waypoint, speed over ground in m/s
        self.last_wp_active = False     # Boolean variable to indicate if the goal is reached, will send to LOS guidance
        self.final_heading_rad = 0.0    # Final heading angle in radians (-pi and pi) when reached the goal, will send to LOS guidance

        self.traj_plan_for_visualization = "" # This is used for visualization purposes, it will not be sent to the FMU
        
        # VARIABLES USED INSIDE THE FMU 
        self.traj_plan = []
        
        # REGISTRATION
        self.register_variable(Real("x_north", causality=Fmi2Causality.input))
        self.register_variable(Real("y_east", causality=Fmi2Causality.input))

        self.register_variable(Boolean("ned_coord", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("lat_reference", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("lon_reference", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("r_accept", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Integer("nu_of_wps", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))

        self.register_variable(Real("wp_01_y_east_lon", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_01_x_north_lat", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_01_sog", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))

        self.register_variable(Real("wp_02_y_east_lon", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_02_x_north_lat", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_02_sog", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))

        self.register_variable(Real("wp_03_y_east_lon", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_03_x_north_lat", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_03_sog", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))

        self.register_variable(Real("wp_04_y_east_lon", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_04_x_north_lat", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_04_sog", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))

        self.register_variable(Real("wp_05_y_east_lon", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_05_x_north_lat", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_05_sog", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))

        self.register_variable(Real("wp_06_y_east_lon", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_06_x_north_lat", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("wp_06_sog", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("final_heading_deg", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))

        self.register_variable(Real("prev_wp_y_east", causality=Fmi2Causality.output))
        self.register_variable(Real("prev_wp_x_north", causality=Fmi2Causality.output))
        self.register_variable(Real("prev_wp_sog", causality=Fmi2Causality.output))

        self.register_variable(Real("next_wp_y_east", causality=Fmi2Causality.output))
        self.register_variable(Real("next_wp_x_north", causality=Fmi2Causality.output))
        self.register_variable(Real("next_wp_sog", causality=Fmi2Causality.output))
        self.register_variable(Boolean("last_wp_active", causality=Fmi2Causality.output))
        self.register_variable(Real("final_heading_rad", causality=Fmi2Causality.output))

        self.register_variable(String("traj_plan_for_visualization", causality=Fmi2Causality.output, variability=Fmi2Variability.fixed))


    def wrap_to_pi(self, angle):
        # Wraps the angle in radians between [-pi,pi)
        res = np.fmod((angle + 2 * np.pi), (2 * np.pi))
        if res > np.pi:
            res -= 2*np.pi
        return res


    def llh2flat(self, lat, lon, sog):
        """
        Compute cartesian local coordinates (east (m), north (m)) from lon (deg), lat (deg).
        
        Params:
            * lat: Position in latitude [deg]
            * lon: Position in longitude [deg]

        Returns:
            * east: Position, east [m]
            * north: Position, north [m]
        """
        height_ref = 0.0

        # WGS-84 parameters
        a_radius = 6378137  # Semi-major axis (equitorial radius)
        f_factor = 1 / 298.257223563  # Flattening
        e_eccentricity = np.sqrt(2 * f_factor - f_factor**2)  # Earth eccentricity

        d_lon = np.deg2rad(lon) - np.deg2rad(self.lon_reference)
        d_lat = np.deg2rad(lat) - np.deg2rad(self.lat_reference)

        r_n = a_radius / np.sqrt(1 - e_eccentricity**2 * np.sin(np.deg2rad(self.lat_reference)) ** 2)
        r_m = r_n * ((1 - e_eccentricity**2) / (1 - e_eccentricity**2 * np.sin(np.deg2rad(self.lat_reference)) ** 2))

        x = d_lat * (r_m + height_ref)
        y = d_lon * ((r_n + height_ref) * np.cos(np.deg2rad(self.lat_reference)))
        
        return x, y, sog
    
    def flat2llh(self, x_north: float, y_east: float,) -> Tuple[float, float]:
        """ 
        Compute latitude and longitude in degrees from local cartesian coordinates (east (m), north (m)).
        
        Params:
            * x_north: Position, north [m]
            * y_east: Position, east [m]
        
        Returns:
            * lat_deg_ais: Position in latitude [deg]
            * lon_deg_ais: Position in longitude [deg]
        """
        # WGS-84 parameters
        a_radius = 6378137  # Semi-major axis
        f_factor = 1 / 298.257223563  # Flattening
        e_eccentricity = math.sqrt(2 * f_factor - f_factor**2)  # Earth eccentricity

        r_n = a_radius / math.sqrt(1 - e_eccentricity**2 * math.sin(math.radians(self.lat_reference)) ** 2)
        r_m = r_n * ((1 - e_eccentricity**2) / (1 - e_eccentricity**2 * math.sin(math.radians(self.lat_reference)) ** 2))

        d_lat = x_north / r_m  # delta latitude dmu = mu - mu0
        d_lon = y_east / (r_n * math.cos(math.radians(self.lat_reference)))  # delta longitude dl = l - l0

        lat_rad = self.wrap_to_pi(math.radians(self.lat_reference) + d_lat)
        lon_rad = self.wrap_to_pi(math.radians(self.lon_reference) + d_lon)
        
        lat_deg_ais = np.rad2deg(lat_rad)
        lon_deg_ais = np.rad2deg(lon_rad)

        return lat_deg_ais, lon_deg_ais


    def calculate_desired_speed(self, wp_leg_distance, ship_to_wp_distance, prev_speed, next_speed):
        # Ensure the distance to the next waypoint is positive to avoid division by zero
        if ship_to_wp_distance <= 0:
            return next_speed
        # Calculate the interpolation factor based on how far the ship is from the next waypoint
        # This factor will be 1 when the ship is at the starting waypoint and 0 when it reaches the next waypoint
        interpolation_factor = ship_to_wp_distance / wp_leg_distance
        # Calculate the desired speed by interpolating between prev_speed and next_speed
        desired_speed = next_speed + interpolation_factor * (prev_speed - next_speed)
        return desired_speed


    def to_json(self):
        coordinates = []
        for i in range(self.nu_of_wps):
            if self.ned_coord:
                x_north = getattr(self, f"wp_0{i+1}_x_north_lat")
                y_east = getattr(self, f"wp_0{i+1}_y_east_lon")
                lat_deg, lon_deg = self.flat2llh(x_north, y_east)
                coordinates.append([lon_deg, lat_deg])
            else:
                # Directly use lat/lon if ned_coord is False
                lat_deg = getattr(self, f"wp_0{i+1}_x_north_lat")
                lon_deg = getattr(self, f"wp_0{i+1}_y_east_lon")
                coordinates.append([lon_deg, lat_deg])
        
        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
        }

        return json.dumps(geojson, default=lambda x: x.__dict__)


    def exit_initialization_mode(self): 
        self.traj_plan = [(self.wp_01_x_north_lat, self.wp_01_y_east_lon, self.wp_01_sog), 
                          (self.wp_02_x_north_lat, self.wp_02_y_east_lon, self.wp_02_sog),
                          (self.wp_03_x_north_lat, self.wp_03_y_east_lon, self.wp_03_sog),
                          (self.wp_04_x_north_lat, self.wp_04_y_east_lon, self.wp_04_sog),
                          (self.wp_05_x_north_lat, self.wp_05_y_east_lon, self.wp_05_sog),
                          (self.wp_06_x_north_lat, self.wp_06_y_east_lon, self.wp_06_sog)]
        
        self.traj_plan = self.traj_plan[0:self.nu_of_wps]

        # Convert LAT/LON positions to ENU in trajectory plan
        if not self.ned_coord:
            traj_plan_ned = []
            for each in self.traj_plan:
                x, y, sog = self.llh2flat(lat=each[0], lon=each[1], sog=each[2])
                traj_plan_ned.append((x, y, sog))
            self.traj_plan = traj_plan_ned
        
        self.prev_wp_y_east = self.y_east
        self.prev_wp_x_north = self.x_north
        self.prev_wp_sog = self.traj_plan[0][2]

        self.next_wp_x_north = self.traj_plan[0][0]
        self.next_wp_y_east = self.traj_plan[0][1]
        self.next_wp_sog = self.traj_plan[0][2]

        self.traj_plan_for_visualization = self.to_json()


    def do_step(self, current_time: float, step_size: float) -> bool:
        for i in range(len(self.traj_plan)-1):
            dist_to_wp = np.array([self.traj_plan[i][0] - self.x_north, 
                                     self.traj_plan[i][1] - self.y_east])

            if np.linalg.norm(dist_to_wp) <= self.r_accept:
                self.prev_wp_x_north = self.traj_plan[i][0]
                self.prev_wp_y_east = self.traj_plan[i][1]
                self.prev_wp_sog = self.traj_plan[i][2]
        
                self.next_wp_x_north = self.traj_plan[i+1][0]
                self.next_wp_y_east = self.traj_plan[i+1][1]
                self.next_wp_sog = self.traj_plan[i+1][2]
        
        # Gradually reduce desired speed to the last wp desired speed
        #if self.next_wp_x_north == self.traj_plan[-1][0] and self.next_wp_y_east == self.traj_plan[-1][1]:
        #    ship_to_wp_distance = np.linalg.norm(np.array([self.next_wp_x_north - self.x_north, 
        #                                                   self.next_wp_y_east - self.y_east]))
        #    
        #    if ship_to_wp_distance <= self.r_accept:
        #        self.last_wp_active = True
        #        
        #if self.last_wp_active:
        #    #self.next_wp_sog = 0.0
        #    self.final_heading_rad = self.wrap_to_pi(np.deg2rad(self.final_heading_deg))
        
        if self.next_wp_x_north == self.traj_plan[-1][0] and self.next_wp_y_east == self.traj_plan[-1][1]:
            self.last_wp_active = True
            self.final_heading_rad = self.wrap_to_pi(np.deg2rad(self.final_heading_deg))

        self.traj_plan_for_visualization = self.traj_plan_for_visualization
        
        return True 