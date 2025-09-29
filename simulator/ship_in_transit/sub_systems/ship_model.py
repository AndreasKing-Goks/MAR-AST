""" 
This module provides classes to construct the ship model to simulate.
It requires the construction of the ship machinery system to construct the ship model.
"""


import numpy as np
import copy
from collections import defaultdict
from typing import NamedTuple, List

from simulator.ship_in_transit.utils.utils import EulerInt, ShipDraw
from simulator.ship_in_transit.sub_systems.ship_engine import ShipMachineryModel, MachinerySystemConfiguration
from simulator.ship_in_transit.sub_systems.wave_model import WaveModelConfiguration
from simulator.ship_in_transit.sub_systems.current_model import CurrentModelConfiguration
from simulator.ship_in_transit.sub_systems.wind_model import WindModelConfiguration
from simulator.ship_in_transit.sub_systems.controllers import (EngineThrottleFromSpeedSetPoint, 
                                                               ThrottleControllerGains, 
                                                               HeadingBySampledRouteController,
                                                               HeadingControllerGains,
                                                               LosParameters)
from simulator.ship_in_transit.sub_systems.sbmpc import SBMPC

from simulator.ship_in_transit.utils import check_condition


###################################################################################################################
####################################### CONFIGURATION FOR SHIP MODEL ##############################################
###################################################################################################################

class ShipConfiguration(NamedTuple):
    dead_weight_tonnage: float
    coefficient_of_deadweight_to_displacement: float
    bunkers: float
    ballast: float
    length_of_ship: float
    width_of_ship: float
    added_mass_coefficient_in_surge: float
    added_mass_coefficient_in_sway: float
    added_mass_coefficient_in_yaw: float
    mass_over_linear_friction_coefficient_in_surge: float
    mass_over_linear_friction_coefficient_in_sway: float
    mass_over_linear_friction_coefficient_in_yaw: float
    nonlinear_friction_coefficient__in_surge: float
    nonlinear_friction_coefficient__in_sway: float
    nonlinear_friction_coefficient__in_yaw: float


class EnvironmentConfiguration(NamedTuple):
    current_velocity_component_from_north: float
    current_velocity_component_from_east: float
    wind_speed: float
    wind_direction: float
    

class SimulationConfiguration(NamedTuple):
    initial_north_position_m: float
    initial_east_position_m: float
    initial_yaw_angle_rad: float
    initial_forward_speed_m_per_s: float
    initial_sideways_speed_m_per_s: float
    initial_yaw_rate_rad_per_s: float
    integration_step: float
    simulation_time: float
    
    
###################################################################################################################
###################################################################################################################

class BaseShipModel:
    def __init__(
            self, ship_config: ShipConfiguration,
            simulation_config: SimulationConfiguration,
            wave_model_config: WaveModelConfiguration,
            current_model_config: CurrentModelConfiguration,
            wind_model_config: WindModelConfiguration
    ):
        # Set configuration as an attribute
        self.ship_config = ship_config
        self.simulation_config = simulation_config
        self.wave_model_config = wave_model_config
        self.current_model_config = current_model_config
        self.wind_model_config = wind_model_config
        
        payload = 0.9 * (self.ship_config.dead_weight_tonnage - self.ship_config.bunkers)
        lsw = self.ship_config.dead_weight_tonnage / self.ship_config.coefficient_of_deadweight_to_displacement \
              - self.ship_config.dead_weight_tonnage
        self.mass = lsw + payload + self.ship_config.bunkers + self.ship_config.ballast

        self.rho = wave_model_config.rho    # Sea water density
        
        self.l_ship = self.ship_config.length_of_ship  # 80
        self.w_ship = self.ship_config.width_of_ship  # 16.0
        self.t_ship = self.mass / (self.rho * self.l_ship * self.w_ship)    # Ship draft assuming ΣFz=0
        self.x_g = 0
        self.i_z = self.mass * (self.l_ship ** 2 + self.w_ship ** 2) / 12
        
        ## Environment model
        w_min = wave_model_config.minimum_wave_frequency
        w_max = wave_model_config.maximum_wave_frequency
        N_omega = wave_model_config.wave_frequency_discrete_unit_count
        psi_min = wave_model_config.minimum_spreading_angle
        psi_max = wave_model_config.maximum_spreading_angle
        N_psi = wave_model_config.spreading_angle_discrete_unit_count
        
        # Vector for each wave across all frequencies
        self.omega_vec = np.linspace(w_min, w_max, N_omega)
        self.domega = self.omega_vec[1] - self.omega_vec[0]
        
        # Vector for wave numbers
        self.g = 9.81
        self.k_vec = self.omega_vec**2 / self.g
        
        # Vector for each wave across discretized spreading direction
        self.psi_vec = np.linspace(psi_min, psi_max, N_psi)
        self.dpsi = self.psi_vec[1] - self.psi_vec[0]
        
        # Vector for randp, phases for each wave across all frequencies
        self.theta = 2.0 * np.pi * np.random.rand(N_omega, N_psi)    # (Nw, Nd)

        # zero-frequency added mass
        self.x_du, self.y_dv, self.n_dr = self.set_added_mass(self.ship_config.added_mass_coefficient_in_surge,
                                                              self.ship_config.added_mass_coefficient_in_sway,
                                                              self.ship_config.added_mass_coefficient_in_yaw)

        self.t_surge = self.ship_config.mass_over_linear_friction_coefficient_in_surge
        self.t_sway = self.ship_config.mass_over_linear_friction_coefficient_in_sway
        self.t_yaw = self.ship_config.mass_over_linear_friction_coefficient_in_yaw
        self.ku = self.ship_config.nonlinear_friction_coefficient__in_surge  # 2400.0  # non-linear friction coeff in surge
        self.kv = self.ship_config.nonlinear_friction_coefficient__in_sway  # 4000.0  # non-linear friction coeff in sway
        self.kr = self.ship_config.nonlinear_friction_coefficient__in_yaw  # 400.0  # non-linear friction coeff in yaw

        # Initialize states
        self.north = np.float64(self.simulation_config.initial_north_position_m)
        self.east = np.float64(self.simulation_config.initial_east_position_m)
        self.yaw_angle = np.float64(self.simulation_config.initial_yaw_angle_rad)
        self.forward_speed = np.float64(self.simulation_config.initial_forward_speed_m_per_s)
        self.sideways_speed = np.float64(self.simulation_config.initial_sideways_speed_m_per_s)
        self.yaw_rate = np.float64(self.simulation_config.initial_yaw_rate_rad_per_s)

        # Initialize differentials
        self.d_north = 0
        self.d_east = 0
        self.d_yaw = 0
        self.d_forward_speed = 0
        self.d_sideways_speed = 0
        self.d_yaw_rate = 0

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(self.simulation_config.integration_step)
        self.int.set_sim_time(self.simulation_config.simulation_time)

        # Instantiate ship draw plotting
        self.draw = ShipDraw()  # Instantiate the ship drawing class
        self.ship_drawings = [[], []]  # Arrays for storing ship drawing data

        # Setup wind effect on ship
        self.rho_a = 1.2
        self.h_f = 8.0  # mean height above water seen from the front
        self.h_s = 8.0  # mean height above water seen from the side
        self.proj_area_f = self.w_ship * self.h_f  # Projected are from the front
        self.proj_area_l = self.l_ship * self.h_s  # Projected area from the side
        self.cx = 0.5
        self.cy = 0.7
        self.cn = 0.08
        
        # Stop Flags
        self.stop = False
        
        # Record of the initial parameters
        self.record_initial_parameters()

    def set_added_mass(self, surge_coeff, sway_coeff, yaw_coeff):
        ''' Sets the added mass in surge due to surge motion, sway due
            to sway motion and yaw due to yaw motion according to given coeffs.

            args:
                surge_coeff (float): Added mass coefficient in surge direction due to surge motion
                sway_coeff (float): Added mass coefficient in sway direction due to sway motion
                yaw_coeff (float): Added mass coefficient in yaw direction due to yaw motion
            returns:
                x_du (float): Added mass in surge
                y_dv (float): Added mass in sway
                n_dr (float): Added mass in yaw
        '''
        x_du = self.mass * surge_coeff
        y_dv = self.mass * sway_coeff
        n_dr = self.i_z * yaw_coeff
        return x_du, y_dv, n_dr

    def get_wind_force(self, wind_args):
        ''' This method calculates the forces due to the relative
            wind speed, acting on the ship in surge, sway and yaw
            direction.

            :return: Wind force acting in surge, sway and yaw
        '''
        # Unpack wind_args
        wind_speed, wind_dir = wind_args
        
        uw = wind_speed * np.cos(wind_dir - self.yaw_angle)
        vw = wind_speed * np.sin(wind_dir - self.yaw_angle)
        u_rw = uw - self.forward_speed
        v_rw = vw - self.sideways_speed
        gamma_rw = -np.arctan2(v_rw, u_rw)
        wind_rw2 = u_rw ** 2 + v_rw ** 2
        c_x = -self.cx * np.cos(gamma_rw)
        c_y = self.cy * np.sin(gamma_rw)
        c_n = self.cn * np.sin(2 * gamma_rw)
        tau_coeff = 0.5 * self.rho_a * wind_rw2
        tau_u = tau_coeff * c_x * self.proj_area_f
        tau_v = tau_coeff * c_y * self.proj_area_l
        tau_n = tau_coeff * c_n * self.proj_area_l * self.l_ship
        return np.array([tau_u, tau_v, tau_n])
    
    def get_wave_force(self, wave_args):
        # Unpack wave_args
        S_w, D_psi, psi_0 = wave_args
        
        # Component elevation amplitudes
        a_eta = np.sqrt(2.0 * np.outer(S_w, D_psi) * self.domega * self.dpsi)   # (Nw, Nd)
        
        # Get ship speed and heading
        ship_speed = np.sqrt(self.forward_speed**2 + self.sideways_speed**2)
        psi_ship = self.yaw_angle
        
        # Encounter correction [Forward speed effect in Faltinsen (1993)]
        beta = self.psi_vec[None, :] - psi_ship                                                 # (1, Nd)
        omega_e = self.omega_vec[:, None] - self.k_vec[:, None] * ship_speed * np.cos(beta)     # (Nw, 1) - (Nw, 1)*(1, Nd) = (Nw, Nd)
        
        # Approximation of oblique wave
        beta_0 = psi_0 - psi_ship
        A_proj = (self.w_ship * np.cos(beta_0) + self.l_ship * np.sin(beta_0)) * self.t_ship
        
        # Froude-Krylov flat-plate force amplitude
        F0 = self.rho * self.g * a_eta * A_proj
        
        # Direction unit vectors
        cx = np.cos(beta)  # (1, Nd)
        cy = np.sin(beta)  # (1, Nd)
        
        # Fx(t) = sum_{i,j} F0[i,j] * cos(omega_e[i,j]*t + phi[i,j]) * cos(theta_j)
        arg    = omega_e * self.int.dt + self.theta
        cosarg = np.cos(arg)                                                      # (Nw, Nd)
        
        # advance phases by Δt
        self.theta = (self.theta + omega_e * self.int.dt) % (2*np.pi)
        # eta = A cos(omega_e*dt + psi0) = A cos(theta)
        # Discrete step t_k = k * dt
        # Instead of tracking the k, we can advances the theta by:
        # Adding the theta with another omega_e*dt, divide by a full sinusoidal cycle of 2*pi,
        # then get the remain. This remain is the advances of theta within the [0, 2*pi)

        # Component forces along x,y per (i,j)
        Fx_ij = F0 * cosarg * cx   # (Nw, Nd)
        Fy_ij = F0 * cosarg * cy   # (Nw, Nd)

        # ---- Lever arms for yaw moment (about CG) ----
        # Smoothly blend bow/stern (±L/2) and port/stbd (±B/2) with heading weight
        def r_cp_from_beta(beta, L, B):
            wx = np.abs(np.cos(beta))
            wy = np.abs(np.sin(beta))
            wsum = wx + wy + 1e-12  # avoid zero
            rx = 0.5 * L * np.sign(np.cos(beta)) * (wx / wsum)
            ry = 0.5 * B * np.sign(np.sin(beta)) * (wy / wsum)
            return rx, ry

        rx, ry = r_cp_from_beta(beta, self.l_ship, self.w_ship)  # (1, Nd)

        # Yaw moment sum Mz = r_x*Fy - r_y*Fx (sum over freq & dir)
        Mz_t = np.sum(rx * Fy_ij - ry * Fx_ij, axis=(0, 1))                 # scalar

        # Total forces (sum over freq & dir)
        Fx_t = np.sum(Fx_ij, axis=(0, 1))
        Fy_t = np.sum(Fy_ij, axis=(0, 1))

        return np.array([Fx_t, Fy_t, Mz_t])

    def three_dof_kinematics(self):
        ''' Updates the time differientials of the north position, east
            position and yaw angle. Should be called in the simulation
            loop before the integration step.
        '''
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])
        dx = np.dot(self.rotation(), vel)
        self.d_north = dx[0]
        self.d_east = dx[1]
        self.d_yaw = dx[2]

    def rotation(self):
        ''' Specifies the rotation matrix for rotations about the z-axis, such that
            "body-fixed coordinates" = rotation x "North-east-down-fixed coordinates" .
        '''
        return np.array([[np.cos(self.yaw_angle), -np.sin(self.yaw_angle), 0],
                         [np.sin(self.yaw_angle), np.cos(self.yaw_angle), 0],
                         [0, 0, 1]])

    def mass_matrix(self):
        return np.array([[self.mass + self.x_du, 0, 0],
                         [0, self.mass + self.y_dv, self.mass * self.x_g],
                         [0, self.mass * self.x_g, self.i_z + self.n_dr]])

    def coriolis_matrix(self):
        return np.array([[0, 0, -self.mass * (self.x_g * self.yaw_rate + self.sideways_speed)],
                         [0, 0, self.mass * self.forward_speed],
                         [self.mass * (self.x_g * self.yaw_rate + self.sideways_speed),
                          -self.mass * self.forward_speed, 0]])

    def coriolis_added_mass_matrix(self, u_r, v_r):
        return np.array([[0, 0, self.y_dv * v_r],
                        [0, 0, -self.x_du * u_r],
                        [-self.y_dv * v_r, self.x_du * u_r, 0]])

    def linear_damping_matrix(self):
        return np.array([[self.mass / self.t_surge, 0, 0],
                      [0, self.mass / self.t_sway, 0],
                      [0, 0, self.i_z / self.t_yaw]])

    def non_linear_damping_matrix(self):
        return np.array([[self.ku * self.forward_speed, 0, 0],
                       [0, self.kv * self.sideways_speed, 0],
                       [0, 0, self.kr * self.yaw_rate]])

    def three_dof_kinetics(self, env_args=None, *args, **kwargs):
        ''' Calculates accelerations of the ship, as a funciton
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        # Environmental conditions
        if env_args is None:
            wind_force = np.array([0.0, 0.0, 0.0])
            wave_force = np.array([0.0, 0.0, 0.0])
            vel_c = np.array([0.0, 0.0, 0.0])
        else:
            wave_args, current_args, wind_args = env_args
            
            wave_force = wave_args
            
            current_speed, current_dir = current_args
            vel_c = np.array([
                current_speed * np.sin(current_dir),
                current_speed * np.cos(current_dir),
                0.0])
            
            wind_dir, wind_speed = wind_args
            wind_force = self.get_wind_force(wind_dir, 
                                            wind_speed)

        # Assembling state vector
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), vel_c)
        u_r = self.forward_speed - v_c[0]
        v_r = self.sideways_speed - v_c[1]

        # Kinetic equation
        m_inv = np.linalg.inv(self.mass_matrix())
        dx = np.dot(
            m_inv,
            -np.dot(self.coriolis_matrix(), vel)
            - np.dot(self.coriolis_added_mass_matrix(u_r=u_r, v_r=v_r), vel - v_c)
            - np.dot(self.linear_damping_matrix() + self.non_linear_damping_matrix(), vel - v_c)
            + wind_force + wave_force
        )
        self.d_forward_speed = dx[0]
        self.d_sideways_speed = dx[1]
        self.d_yaw_rate = dx[2]

    def update_differentials(self, env_args=None, *args, **kwargs):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        self.three_dof_kinetics(env_args)

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.north = self.int.integrate(x=self.north, dx=self.d_north)
        self.east = self.int.integrate(x=self.east, dx=self.d_east)
        self.yaw_angle = self.int.integrate(x=self.yaw_angle, dx=self.d_yaw)
        self.forward_speed = self.int.integrate(x=self.forward_speed, dx=self.d_forward_speed)
        self.sideways_speed = self.int.integrate(x=self.sideways_speed, dx=self.d_sideways_speed)
        self.yaw_rate = self.int.integrate(x=self.yaw_rate, dx=self.d_yaw_rate)

    def ship_snap_shot(self):
        ''' This method is used to store a map-view snap shot of
            the ship at the given north-east position and heading.
            It uses the ShipDraw-class. To plot a map view of the
            n-th ship snap-shot, use:

            plot(ship_drawings[1][n], ship_drawings[0][n])
        '''
        x, y = self.draw.local_coords()
        x_ned, y_ned = self.draw.rotate_coords(x, y, self.yaw_angle)
        x_ned_trans, y_ned_trans = self.draw.translate_coords(x_ned, y_ned, self.north, self.east)
        self.ship_drawings[0].append(x_ned_trans)
        self.ship_drawings[1].append(y_ned_trans)
        
    def record_initial_parameters(self):
        '''
        Internal method to take a record of internal attributes after __init__().
        This record will be used to reset the model later without re-instantiation.
        '''
        self._initial_parameters = {
            key: copy.deepcopy(self.__dict__[key])
            for key in ['mass', 'i_z', 'x_du', 'y_dv', 'n_dr',
                        't_surge', 't_sway', 't_yaw',
                        'ku', 'kv', 'kr',
                        'north', 'east', 'yaw_angle',
                        'forward_speed', 'sideways_speed', 'yaw_rate',
                        'd_north', 'd_east', 'd_yaw',
                        'd_forward_speed', 'd_sideways_speed', 'd_yaw_rate',
                        'proj_area_f', 'proj_area_l',
                        'omega_vec', 'domega', 'k_vec', 'psi_vec', 'dpsi', 'theta',
                        'ship_drawings', 'stop']
            }

    def reset(self):
        '''
        Resets the internal state of the ship to its initial record.
        Should be called at the beginning of each episode.
        '''
        for key, value in self._initial_parameters.items():
            setattr(self, key, copy.deepcopy(value))

        # Re-create integrator and drawer
        self.int = EulerInt()
        self.int.set_dt(self.simulation_config.integration_step)
        self.int.set_sim_time(self.simulation_config.simulation_time)
        
        # Reset the environment model
        self.wave_model.reset()
        self.current_model.reset()
        self.wind_model.reset()

        self.draw = ShipDraw()

###################################################################################################################
########################## DESCENDANT CLASS BASED ON PARENT CLASS "BaseShipModel" #################################
###################################################################################################################

class ShipModel(BaseShipModel):
    ''' Creates a ship model object that can be used to simulate a ship in transit, used 
        particularly for stress testing purposes. Wave, wind and current environment load 
        is also included.

        The ships model is propelled by a single propeller and steered by a rudder.
        The propeller is powered by either the main engine, an auxiliary motor
        referred to as the hybrid shaft generator, or both. The model contains the
        following states:
        - North position of ship
        - East position of ship
        - Yaw angle (relative to north axis)
        - Surge velocity (forward)
        - Sway velocity (sideways)
        - Yaw rate
        - Propeller shaft speed

        Simulation results are stored in the instance variable simulation_results
    '''
    def __init__(self, ship_config: ShipConfiguration, 
                 simulation_config: SimulationConfiguration,
                 wave_model_config: WaveModelConfiguration,
                 current_model_config: CurrentModelConfiguration,
                 wind_model_config: WindModelConfiguration,
                 machinery_config: MachinerySystemConfiguration,
                 throttle_controller_gain: ThrottleControllerGains,
                 heading_controller_gain: HeadingControllerGains,
                 los_parameters: LosParameters,
                 name_tag: str,
                 route_name,
                 desired_speed,
                 engine_steps_per_time_step,
                 initial_propeller_shaft_speed_rad_per_s,
                 map_obj=None,
                 colav_mode=None):
        super().__init__(ship_config, simulation_config, wave_model_config, current_model_config, wind_model_config)
        
        if map_obj is not None:
            self.map_obj = map_obj
        
        self.ship_machinery_model = ShipMachineryModel(
            machinery_config=machinery_config,
            initial_propeller_shaft_speed_rad_per_sec=initial_propeller_shaft_speed_rad_per_s,
            time_step=self.int.dt/engine_steps_per_time_step
        )
        self.throttle_controller = EngineThrottleFromSpeedSetPoint(
            gains=throttle_controller_gain,
            max_shaft_speed=self.ship_machinery_model.shaft_speed_max,
            time_step=self.int.dt,
            initial_shaft_speed_integral_error=114
        )
        self.auto_pilot = HeadingBySampledRouteController(
            route_name=route_name,
            heading_controller_gains=heading_controller_gain,
            los_parameters=los_parameters,
            time_step=self.int.dt,
            max_rudder_angle=np.rad2deg(machinery_config.max_rudder_angle_degrees)
        )
        
        # Ship desired speed
        self.desired_speed = desired_speed
        self.init_desired_speed = self.desired_speed
        
        # Get stop_info
        self.stop_info = {
            'grounding_failure' : False,
            'navigation_failure': False,
            'reaches_endpoint'  : False,
            'outside_horizon'   : False,
            'power_overload'    : False
        }
        
        # Get the collision info
        self.colav_mode = colav_mode
        self.colav_active = False
        
        # Scenario-Based Model Predictive Controller
        self.sbmpc = SBMPC(tf=1000, dt=20)
        
        # Ship name
        self.name_tag = name_tag
        
        # Default dictionary for simulation results
        self.simulation_results = defaultdict(list)

    def three_dof_kinetics(self, 
                           thrust_force=None, 
                           rudder_angle=None, 
                           env_args=None,
                           *args, 
                           **kwargs):
        ''' Calculates accelerations of the ship, as a function
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        # Environmental conditions
        if env_args is None:
            wind_force = np.array([0.0, 0.0, 0.0])
            wave_force = np.array([0.0, 0.0, 0.0])
            vel_c      = np.array([0.0, 0.0, 0.0])
        else:
            wave_args, current_args, wind_args = env_args
            
            wave_force = self.get_wave_force(wave_args)
            wind_force = self.get_wind_force(wind_args)
            
            current_speed, current_dir = current_args
            vel_c = np.array([
                current_speed * np.sin(current_dir),
                current_speed * np.cos(current_dir),
                0.0])

        # Forces acting (replace zero vectors with suitable functions)
        f_rudder_v, f_rudder_r = self.rudder(rudder_angle, vel_c)
        ctrl_force = np.array([thrust_force, f_rudder_v, f_rudder_r])
        
        # assembling state vector
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), vel_c)
        u_r = self.forward_speed - v_c[0]
        v_r = self.sideways_speed - v_c[1]

        # Kinetic equation
        m_inv = np.linalg.inv(self.mass_matrix())
        dx = np.dot(
            m_inv,
            -np.dot(self.coriolis_matrix(), vel)
            - np.dot(self.coriolis_added_mass_matrix(u_r=u_r, v_r=v_r), vel - v_c)
            - np.dot(self.linear_damping_matrix() + self.non_linear_damping_matrix(), vel - v_c)
            + wind_force + wave_force + ctrl_force)
        self.d_forward_speed = dx[0]
        self.d_sideways_speed = dx[1]
        self.d_yaw_rate = dx[2]

    def rudder(self, delta, vel_c):
        ''' This method takes in the rudder angle and returns
            the force i sway and yaw generated by the rudder.

            args:
            delta (float): The rudder angle in radians

            returs:
            v_force (float): The force in sway-direction generated by the rudder
            r_force (float): The yaw-torque generated by the rudder
        '''
        u_c = np.dot(np.linalg.inv(self.rotation()), vel_c)[0]
        v_force = -self.ship_machinery_model.c_rudder_v * delta * (self.forward_speed - u_c)
        r_force = -self.ship_machinery_model.c_rudder_r * delta * (self.forward_speed - u_c)
        return v_force, r_force

    def update_differentials(self, 
                             engine_throttle=None, 
                             rudder_angle=None, 
                             env_args=None,
                             *args, 
                             **kwargs):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        for _ in range(int(self.int.dt / self.ship_machinery_model.int.dt)):
            self.ship_machinery_model.update_shaft_equation(engine_throttle)
        self.three_dof_kinetics(thrust_force=self.ship_machinery_model.thrust(), 
                                rudder_angle=rudder_angle,
                                env_args=env_args)

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.north = self.int.integrate(x=self.north, dx=self.d_north)
        self.east = self.int.integrate(x=self.east, dx=self.d_east)
        self.yaw_angle = self.int.integrate(x=self.yaw_angle, dx=self.d_yaw)
        self.forward_speed = self.int.integrate(x=self.forward_speed, dx=self.d_forward_speed)
        self.sideways_speed = self.int.integrate(x=self.sideways_speed, dx=self.d_sideways_speed)
        self.yaw_rate = self.int.integrate(x=self.yaw_rate, dx=self.d_yaw_rate)
        self.ship_machinery_model.integrate_differentials()
    
    def store_simulation_data(self, load_perc, rudder_angle, e_ct, e_psi):
        load_perc_me, load_perc_hsg = self.ship_machinery_model.load_perc(load_perc)
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.north)
        self.simulation_results['east position [m]'].append(self.east)
        self.simulation_results['yaw angle [deg]'].append(self.yaw_angle * 180 / np.pi)
        self.simulation_results['rudder angle [deg]'].append(rudder_angle * 180 / np.pi)
        self.simulation_results['forward speed [m/s]'].append(self.forward_speed)
        self.simulation_results['sideways speed [m/s]'].append(self.sideways_speed)
        self.simulation_results['yaw rate [deg/sec]'].append(self.yaw_rate * 180 / np.pi)
        self.simulation_results['propeller shaft speed [rpm]'].append(self.ship_machinery_model.omega * 30 / np.pi)
        self.simulation_results['commanded load fraction me [-]'].append(load_perc_me)
        self.simulation_results['commanded load fraction hsg [-]'].append(load_perc_hsg)

        load_data = self.ship_machinery_model.mode.distribute_load(
            load_perc=load_perc, hotel_load=self.ship_machinery_model.hotel_load
        )
        self.simulation_results['power me [kw]'].append(load_data.load_on_main_engine / 1000)
        self.simulation_results['available power me [kw]'].append(
            self.ship_machinery_model.mode.main_engine_capacity / 1000 
        )
        self.simulation_results['power electrical [kw]'].append(load_data.load_on_electrical / 1000)
        self.simulation_results['available power electrical [kw]'].append(
        self.ship_machinery_model.mode.electrical_capacity / 1000
        )
        self.simulation_results['power [kw]'].append((load_data.load_on_electrical
                                                    + load_data.load_on_main_engine) / 1000)
        self.simulation_results['propulsion power [kw]'].append(
            (load_perc * self.ship_machinery_model.mode.available_propulsion_power) / 1000)
        rate_me, rate_hsg, cons_me, cons_hsg, cons = self.ship_machinery_model.fuel_consumption(load_perc)
        self.simulation_results['fuel rate me [kg/s]'].append(rate_me)
        self.simulation_results['fuel rate hsg [kg/s]'].append(rate_hsg)
        self.simulation_results['fuel rate [kg/s]'].append(rate_me + rate_hsg)
        self.simulation_results['fuel consumption me [kg]'].append(cons_me)
        self.simulation_results['fuel consumption hsg [kg]'].append(cons_hsg)
        self.simulation_results['fuel consumption [kg]'].append(cons)
        self.simulation_results['motor torque [Nm]'].append(self.ship_machinery_model.main_engine_torque(load_perc))
        self.simulation_results['thrust force [kN]'].append(self.ship_machinery_model.thrust() / 1000)
        self.simulation_results['cross track error [m]'].append(e_ct)
        self.simulation_results['heading error [deg]'].append(e_psi)
        
    ## ADDITIONAL ##
    def store_last_simulation_data(self):
        '''
        Stores the last known state repeatedly when the ship has stopped moving.
        '''
        if not self.simulation_results['time [s]']:
            raise RuntimeError("No simulation data to repeat — ship has not run yet.")

        # Just use the last known values from the simulation results
        self.simulation_results['time [s]'].append(self.int.time)
        for key in self.simulation_results:
            if key != 'time [s]':
                last_value = self.simulation_results[key][-1]
                self.simulation_results[key].append(last_value)
                
    def evaluate_ship_condition(self):
        '''
        Evaluate the ship condition to determine the stop flags.
        '''
        # Only evaluate this condition when the map_obj is exists
        if self.map_obj is not None:
            if check_condition.is_grounding(map_obj=self.map_obj,
                                            pos=[self.north, self.east],
                                            ship_length=self.l_ship):
                self.stop_info['grounding_failure'] = True
                self.stop = True
                print(self.name_tag, ' in ', self.ship_machinery_model.operating_mode, ' mode experiences grounding.')
            
            if check_condition.is_pos_outside_horizon(map_obj=self.map_obj,
                                                pos=[self.north, self.east],
                                                ship_length=self.l_ship):
                self.stop_info['outside_horizon'] = True
                self.stop = True
                print(self.name_tag, ' in ', self.ship_machinery_model.operating_mode, ' mode is outside the map horizon.')
            
        if check_condition.is_ship_navigation_failure(e_ct=self.auto_pilot.navigate.e_ct,
                                                      e_tol=500):
            self.stop_info['navigation_failure'] = True
            self.stop = True
            print(self.name_tag, ' experiences navigational failure.')
        
        if check_condition.is_reaches_endpoint(route_end=[self.auto_pilot.navigate.north[-1], self.auto_pilot.navigate.east[-1]], 
                                               pos=[self.north, self.east], 
                                               arrival_radius=250):
            self.stop_info['reaches_endpoint'] = True
            self.stop = True
            print(self.name_tag, ' in ', self.ship_machinery_model.operating_mode, ' mode reaches its final destination.')
        
        if len(self.simulation_results['power me [kw]']) > 0:
            if self.ship_machinery_model.operating_mode in ('PTO', 'MEC'):
                power = self.simulation_results['power me [kw]'][-1]
                available_power=self.simulation_results['available power me [kw]'][-1]
            elif self.ship_machinery_model.operating_mode == 'PTI':
                power = self.simulation_results['power electrical [kw]'][-1]
                available_power=self.simulation_results['available power electrical [kw]'][-1]
            
            if check_condition.is_power_overload(power=power, 
                                                 available_power=available_power):
                    self.stop_info['power_overload'] = True
                    self.stop = True
                    print(self.name_tag, ' in ', self.ship_machinery_model.operating_mode, ' mode experiences power overloading.')
        
    
    def step(self, env_args=None, asset_infos=None):
        ''' 
            The method is used for stepping up the simulator for the ship asset
            
            env_args: [wave_args, current_args, wind_args]
            Arguments needed to compute the environmental loads, where:
            - wave_args     : [S_w, D_psi]  # Spectrum value and Spreading factor
            - current_args  : [current_speed, current_dir]
            - wind_args     : [wind_speed, wind_dir]
            
        '''          
        # Measure ship position and speed
        north_position = self.north
        east_position = self.east
        heading = self.yaw_angle
        measured_shaft_speed = self.ship_machinery_model.omega
        measured_speed = np.sqrt(self.forward_speed**2 + self.sideways_speed**2)
        
        # Keep it, even when sbmpc is disabled
        speed_factor, desired_heading_offset = 1.0, 0.0

        ## COLLISION AVOIDANCE - SBMPC
        ####################################################################################################
        if self.colav_mode == 'sbmpc' and asset_infos is not None:
            # Get desired heading and speed for collav
            self.next_wpt, self.prev_wpt = self.auto_pilot.navigate.next_wpt(self.auto_pilot.next_wpt, north_position, east_position)
            chi_d = self.auto_pilot.navigate.los_guidance(self.auto_pilot.next_wpt, north_position, east_position)
            u_d = self.desired_speed

            # Get OS state required for SBMPC
            os_state = np.array([self.east,            # x
                                 self.north,           # y
                                -self.yaw_angle,       # Same reference angle but clockwise positive
                                 self.forward_speed,   # u
                                 self.sideways_speed,  # v
                                 self.yaw_rate         # Unused
                                ])
            
            do_list = []
            for i, asset_info in enumerate(asset_infos):
                if asset_info.name_tag != self.name_tag:  # skip self
                    # print(asset_info.name_tag)
                    do_list.append(
                                        (
                    i, 
                    np.array([
                        asset_info.current_east,
                        asset_info.current_north,
                        -asset_info.current_yaw_angle,
                        asset_info.forward_speed,
                        asset_info.sideways_speed
                    ]),
                    None,
                    asset_info.ship_length,
                    asset_info.ship_width
                )
                    )
            
            speed_factor, desired_heading_offset = self.sbmpc.get_optimal_ctrl_offset(
                    u_d=u_d,
                    chi_d=-chi_d,
                    os_state=os_state,
                    do_list=do_list
                )
            
            # if self.sbmpc.is_stephen_useful():
            #     print(self.name_tag, ' COLAV system activated')
            # Note: self.sbmpc.is_stephen_useful() -> bool can be used to know whether or not the SBMPC colav algorithm is currently active
        #################################################################################################### 
        
        # Evaluate the ship condition. If the ship stopped, immediately return
        self.evaluate_ship_condition()
        if self.stop is True:
            return
        
        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading,
            desired_heading_offset=desired_heading_offset
        )

        throttle = self.throttle_controller.throttle(
            speed_set_point = self.desired_speed * speed_factor,
            measured_speed = measured_speed,
            measured_shaft_speed = measured_shaft_speed,
        )
        
        # Update and integrate differential equations for current time step
        self.store_simulation_data(throttle, 
                                   rudder_angle,
                                   self.auto_pilot.get_cross_track_error(),
                                   self.auto_pilot.get_heading_error())
        self.update_differentials(engine_throttle=throttle, 
                                  rudder_angle=rudder_angle, 
                                  env_args=env_args)
        self.integrate_differentials()
        
        # Step up the simulator
        self.int.next_time()
    
    def reset(self):
        # Call the reset method from the parent class
        super().reset()
        
        # Reset the subsystem
        self.ship_machinery_model.reset()
        self.throttle_controller.reset()
        self.auto_pilot.reset()
        
        # Reset the desired speed
        self.desired_speed = self.init_desired_speed
        
        # Reset the collision info
        self.colav_active = False
        
        # Reset stop_info
        self.stop_info = {
            'grounding_failure' : False,
            'navigation_failure': False,
            'reaches_endpoint'  : False,
            'outside_horizon'   : False,
            'power_overload'    : False
        }
        
        #  Also reset the results and draws container
        self.simulation_results = defaultdict(list)
