import numpy as np

class SurfaceCurrent:
    def __init__(self, init_vel, mu_vel, init_dir, mu_dir, sigma_vel=0.1, sigma_dir=0.1, seed=None, dt=30):
        # Initialize state variables (velocity magnitude and direction)
        self.vel = init_vel         # initial current velocity magnitude [m/s]
        self.mu_vel = mu_vel        # decay rate for velocity (Gauss–Markov)

        self.dir = init_dir         # initial current direction [rad]
        self.mu_dir = mu_dir        # decay rate for direction (Gauss–Markov)

        # Standard deviations for the white noise inputs
        self.sigma_vel = sigma_vel  # noise strength for velocity
        self.sigma_dir = sigma_dir  # noise strength for direction

        self.dt = dt                # timestep size [s]
        
        # Random seed
        if seed is not None:
            np.random.seed(seed)
        
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