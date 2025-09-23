import numpy as np
import copy
from typing import NamedTuple

class WindModelConfiguration(NamedTuple):
    initial_mean_wind_velocity: float
    mean_wind_velocity_decay_rate: float
    mean_wind_velocity_standard_deviation: float
    initial_wind_direction: float
    wind_direction_decay_rate: float
    wind_direction_standard_deviation: float
    minimum_mean_wind_velocity: float
    maximum_mean_wind_velocity: float
    minimum_wind_gust_frequency: float
    maximum_wind_gust_frequency: float
    wind_gust_frequency_discrete_unit_count: int
    clip_speed_nonnegative: bool
    kappa_parameter: float  
    U10: int                                        # 1 hour mean wind speed at 10 m elevation
    wind_evaluation_height: float
    timestep_size: float

class NORSOKWindModel:
    """
    Mean wind:  Ȗ + μ Ȗ = w     (Gauss–Markov, Euler discretized)
    Gust:       U_g(t) = sum_i sqrt(2 S(f_i) Δf) cos(2π f_i t + φ_i)
    Direction:  ψ̇ + μψ ψ = wψ   (Gauss–Markov)
    """
    def __init__(self, config:WindModelConfiguration, seed=None):

        # stochastic/mean params
        self.mu_Ubar = config.mean_wind_velocity_decay_rate
        self.mu_dir = config.wind_direction_decay_rate
        self.sigma_Ubar = config.mean_wind_velocity_standard_deviation
        self.sigma_dir = config.wind_direction_standard_deviation
        self.U_min = config.minimum_mean_wind_velocity
        self.U_max = config.maximum_mean_wind_velocity
        self.dt = config.timestep_size

        # environment / spectrum params
        self.U10 = config.U10
        self.z = config.wind_evaluation_height
        self.kappa = config.kappa_parameter

        # frequency grid
        self.f_min = config.minimum_wind_gust_frequency
        self.f_max = config.maximum_wind_gust_frequency
        self.N_f = config.wind_gust_frequency_discrete_unit_count
        self.f = np.linspace(self.f_min, self.f_max, self.N_f)   # Hz
        self.df = self.f[1] - self.f[0]

        # mean wind at height z (log law used in Fossen around Harris example)
        if config.initial_mean_wind_velocity is None:
            z0 = 10.0 * np.exp(-2.0/(5.0*np.sqrt(self.kappa)))
            self.Ubar = self.U10 * (2.5*np.sqrt(self.kappa)) * np.log(self.z / z0)
        else:
            self.Ubar = float(config.initial_mean_wind_velocity)

        # direction state
        self.dir = float(config.initial_wind_direction)

        # ---- precompute gust amplitudes & phases (fixed), keep running angles ----
        S = self._norsok(self.f)
        self.a = np.sqrt(2.0 * S * self.df)                  # (N_f,)
        self.phi0 = 2*np.pi*np.random.rand(self.N_f)              # fixed phases
        self.theta = self.phi0.copy()                        # running angles

        self.clip_speed_nonnegative =  config.clip_speed_nonnegative
        
        # Random seed
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Record of the initial parameters
        self.record_initial_parameters()

    # ----- spectra -----
    def _norsok(self, f):
        # S(f) = 320 (U10/10)^2 (z/10)^0.45 / (1 + x^n)^(5/(3n))
        # x = 172 f (z/10)^(2/3) (U10/10)^(-3/4), n = 0.468
        n = 0.468
        x = 172.0 * f * (self.z/10.0)**(2.0/3.0) * (self.U10/10.0)**(-3.0/4.0)
        return 320.0 * (self.U10/10.0)**2 * (self.z/10.0)**0.45 / (1.0 + x**n)**(5.0/(3.0*n))

    def compute_wind_mean_velocity(self):
        # Ȗ_{k+1} = Ȗ_k + dt(-μ Ȗ_k + w), w ~ N(0, σ^2/dt)
        w = np.random.normal(0.0, self.sigma_Ubar/np.sqrt(self.dt))
        self.Ubar += self.dt * (-self.mu_Ubar * self.Ubar + w)
        self.Ubar = np.clip(self.Ubar, self.U_min, self.U_max)
        return self.Ubar

    def compute_wind_gust(self):
        # advance phases by 2π f dt and sum components
        self.theta += 2*np.pi*self.f*self.dt
        return np.sum(self.a * np.cos(self.theta))

    def compute_wind_direction(self):
        w = np.random.normal(0.0, self.sigma_dir/np.sqrt(self.dt))
        self.dir += self.dt * (-self.mu_dir * self.dir + w)
        # wrap to [-pi, pi]
        self.dir = (self.dir + np.pi) % (2*np.pi) - np.pi
        return self.dir

    def get_wind_vel_and_dir(self):
        """Advance model by one dt. Returns (U_speed, direction)."""
        Ubar = self.compute_wind_mean_velocity()
        Ug   = self.compute_wind_gust()
        U_w    = Ubar + Ug
        if self.clip_speed_nonnegative:
            U_w = max(0.0, U_w)
        psi_w  = self.compute_wind_direction()
        return U_w, psi_w
    
    def record_initial_parameters(self):
        '''
        Internal method to take a record of internal attributes after __init__().
        This record will be used to reset the model later without re-instantiation.
        '''
        self._initial_parameters = {
            key: copy.deepcopy(self.__dict__[key])
            for key in ['Ubar', 'mu_Ubar', 'mu_dir', 'sigma_Ubar', 
                        'sigma_dir', 'U_min', 'U_max', 'dt', 'U10', 'z', 
                        'kappa', 'f_min', 'f_max', 'N_f', 'f', 'df', 'a',
                        'seed']
            }

    def reset(self):
        for key, value in self._initial_parameters.items():
            setattr(self, key, copy.deepcopy(value))
        self.phi0 = 2*np.pi*np.random.rand(self.N_f)         # fixed phases
        self.theta = self.phi0.copy()                        # running angles
        
        