import numpy as np
import matplotlib.pyplot as plt

class SurfaceCurrent:
    def __init__(self, init_vel, mu_vel, init_dir, mu_dir, sigma_vel=0.1, sigma_dir=0.05, seed=None, dt=30, clip_speed_nonnegative=True):
        # Initialize state variables (velocity magnitude and direction)
        self.vel = init_vel         # initial current velocity magnitude [m/s]
        self.mu_vel = mu_vel        # decay rate for velocity (Gauss–Markov)

        self.dir = init_dir         # initial current direction [rad]
        self.mu_dir = mu_dir        # decay rate for direction (Gauss–Markov)

        # Standard deviations for the white noise inputs
        self.sigma_vel = sigma_vel  # noise strength for velocity
        self.sigma_dir = sigma_dir  # noise strength for direction

        self.dt = dt                # timestep size [s]
        
        self.clip_speed_nonnegative = True
        
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
        if self.clip_speed_nonnegative:
            U_c = max(0.0, U_c)
        psi_c = self.compute_current_direction()
        
        return U_c, psi_c
    

# -------------------------------
# Example simulation setup
# -------------------------------
init_vel = 0            # initial current velocity [m/s]
mu_vel = 0.05           # decay rate for velocity

init_dir = np.deg2rad(-90)  # initial direction: -90 deg = South
mu_dir = 0.05               # decay rate for direction

current_model = SurfaceCurrent(init_vel, mu_vel, init_dir, mu_dir)

T = 6000                       # total simulation time [s]
Nt = int(T/current_model.dt)   # number of steps

t = np.arange(Nt) * current_model.dt   # time vector
U_c = np.zeros(Nt)                     # store velocity time history
psi_c = np.zeros(Nt)                     # store direction time history

# Time stepping loop: update current model at each timestep
for k in range(Nt):
    U_c[k], psi_c[k] = current_model.get_current_vel_and_dir()
    

# -------------------------------
# Plot results
# -------------------------------

# Plot velocity magnitude over time
plt.figure()
plt.plot(t, U_c, label='Current velocity [m/s]')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.title('Current Velocity')
plt.grid(True)

# Plot current direction (converted to degrees) over time
plt.figure()
plt.plot(t, np.rad2deg(psi_c), label='Current direction [deg]')
plt.xlabel('Time [s]')
plt.ylabel('Direction [deg]')
plt.legend()
plt.title('Current Direction')
plt.grid(True)

plt.show()
