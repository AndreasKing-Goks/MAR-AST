import numpy as np
import copy
from typing import NamedTuple

class CurrentModelConfiguration(NamedTuple):
    initial_current_velocity: float
    current_velocity_standard_deviation: float
    current_velocity_decay_rate: float
    initial_current_direction: float
    current_direction_standard_deviation: float
    current_direction_decay_rate: float
    timestep_size: float

class SurfaceCurrent:
    def __init__(self, config:CurrentModelConfiguration, seed=None):
        # Initialize state variables (velocity magnitude and direction)
        self.vel = config.initial_current_velocity                    # initial current velocity magnitude [m/s]
        self.mu_vel = config.current_velocity_decay_rate              # decay rate for velocity (Gauss–Markov)

        self.dir = config.initial_current_direction                   # initial current direction [rad]
        self.mu_dir = config.current_direction_decay_rate             # decay rate for direction (Gauss–Markov)

        # Standard deviations for the white noise inputs
        self.sigma_vel = config.current_velocity_standard_deviation   # noise strength for velocity
        self.sigma_dir = config.current_direction_standard_deviation  # noise strength for direction

        self.dt = config.timestep_size                                # timestep size [s]
        
        # Random seed
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Record of the initial parameters
        self.record_initial_parameters()
        
    def compute_current_velocity(self):
        # Generate Gaussian white noise for velocity.
        # Scale by 1/sqrt(dt) so variance is consistent with continuous-time noise
        w = np.random.normal(0, self.sigma_vel / np.sqrt(self.dt))  
        
        # Update velocity using Euler discretization of: Vdot + mu*V = w
        self.vel = self.vel + self.dt * (-self.mu_vel * self.vel + w)
        
        return self.vel
    
    def compute_current_direction(self):
        # Generate Gaussian white noise for direction
        w = np.random.normal(0, self.sigma_dir / np.sqrt(self.dt))  
        
        # Update direction using Euler discretization of: ψdot + mu*ψ = w
        self.dir = self.dir + self.dt * (-self.mu_dir * self.dir + w)
        
        # Wrap the direction angle back into [-pi, pi]
        self.dir = (self.dir + np.pi) % (2*np.pi) - np.pi
        
        return self.dir
    
    def get_current_vel_and_dir(self):
        # Update both velocity and direction and return them
        U_c = self.compute_current_velocity()
        psi_c = self.compute_current_direction()
        
        return U_c, psi_c
    
    def record_initial_parameters(self):
        '''
        Internal method to take a record of internal attributes after __init__().
        This record will be used to reset the model later without re-instantiation.
        '''
        self._initial_parameters = {
            key: copy.deepcopy(self.__dict__[key])
            for key in ['vel', 'mu_vel', 'dir', 'mu_dir',
                        'sigma_vel', 'sigma_dir', 'dt',
                        'seed']
            }

    def reset(self):
        for key, value in self._initial_parameters.items():
            setattr(self, key, copy.deepcopy(value))