from pathlib import Path
import sys

## PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from simulator.ship_in_transit.sub_systems.wind_model import NORSOKWindModel, WindModelConfiguration

wind_model_config = WindModelConfiguration(
    initial_mean_wind_velocity=None,                    # Set to None to use a mean wind component
    mean_wind_velocity_decay_rate=0.001,
    mean_wind_velocity_standard_deviation=0.025,
    initial_wind_direction=np.deg2rad(45.0),
    wind_direction_decay_rate=0.001,
    wind_direction_standard_deviation=0.03,
    minimum_mean_wind_velocity=0.0,
    maximum_mean_wind_velocity=100.0,
    minimum_wind_gust_frequency=0.06,
    maximum_wind_gust_frequency=0.4,
    wind_gust_frequency_discrete_unit_count=100,
    clip_speed_nonnegative=True,
    kappa_parameter=0.0026,
    U10=2.5,
    wind_evaluation_height=5.0,
    timestep_size=0.5
)
    
# NORSOK gust, z=20 m, U10=12 m/s
wind_model = NORSOKWindModel(wind_model_config)

T = 3600
Nt = int(T/wind_model.dt)
t  = np.arange(Nt)*wind_model.dt
U_w  = np.zeros(Nt); psi_w = np.zeros(Nt)

Ubar_mean = None
dir_mean = None

Ubar_mean = 1.0
dir_mean = np.deg2rad(-90)

for k in range(Nt):
    U_w[k], psi_w[k] = wind_model.get_wind_vel_and_dir(Ubar_mean, dir_mean)

# -------------------------------
# Plot results
# -------------------------------

# Plot velocity magnitude over time
plt.figure()
plt.plot(t, U_w, label='Wind velocity [m/s]')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.title('Wind Velocity')
plt.grid(True)

# Plot current direction (converted to degrees) over time
plt.figure()
plt.plot(t, np.rad2deg(psi_w), label='Wind direction [deg]')
plt.xlabel('Time [s]')
plt.ylabel('Wind [deg]')
plt.legend()
plt.title('Wind Direction')
plt.grid(True)

plt.show()