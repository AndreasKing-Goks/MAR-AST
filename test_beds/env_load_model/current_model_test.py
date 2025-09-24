from pathlib import Path
import sys

## PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from simulator.ship_in_transit.sub_systems.current_model import SurfaceCurrent, CurrentModelConfiguration

# -------------------------------
# Example simulation setup
# -------------------------------

current_model_config = CurrentModelConfiguration(
    initial_current_velocity=5,
    current_velocity_standard_deviation=0.05,
    current_velocity_decay_rate=0.0025,
    initial_current_direction=np.deg2rad(-45),
    current_direction_standard_deviation=0.05,
    current_direction_decay_rate=0.005,
    timestep_size=0.5
)

current_model = SurfaceCurrent(current_model_config)

T = 6000                       # total simulation time [s]
Nt = int(T/current_model.dt)   # number of steps

t = np.arange(Nt) * current_model.dt   # time vector
U_c = np.zeros(Nt)                     # store velocity time history
psi_c = np.zeros(Nt)                     # store direction time history

vel_mean, dir_mean = 2, np.deg2rad(45)

# Time stepping loop: update current model at each timestep
for k in range(Nt):
    U_c[k], psi_c[k] = current_model.get_current_vel_and_dir(vel_mean, dir_mean)
    

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
