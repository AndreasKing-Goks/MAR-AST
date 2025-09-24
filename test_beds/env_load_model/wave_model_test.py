from pathlib import Path
import sys

## PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

from simulator.ship_in_transit.sub_systems.wave_model import JONSWAPWaveModel, WaveModelConfiguration

ship_speed = 6.0
psi_ship = np.deg2rad(0)
ship_length = 80
ship_breadth = 10
ship_draft = 8
Hs = 0.3
Tp = 7.5
psi_0 = np.deg2rad(45)
rho=1025.0

wave_model_config = WaveModelConfiguration(
    minimum_wave_frequency=0.4,
    maximum_wave_frequency=2.5,
    wave_frequency_discrete_unit_count=50,
    minimum_spreading_angle=-np.pi/2,
    maximum_spreading_angle=np.pi/2,
    spreading_angle_discrete_unit_count=10,
    spreading_coefficient=1,
    rho=rho,
    timestep_size=0.5
)
        
wave_model = JONSWAPWaveModel(config=wave_model_config)
    
T = 300
Nt = int(T/wave_model.dt)

t = np.arange(Nt) * wave_model.dt
Fx = np.zeros(Nt)
Fy = np.zeros(Nt)
M  = np.zeros(Nt)

for k in range(Nt):
    
    wave_force = wave_model.get_direct_wave_force(ship_speed, psi_ship, ship_length, ship_breadth, ship_draft, Hs, Tp, psi_0)
    Fx[k] = wave_force[0]
    Fy[k] = wave_force[1]
    M[k]  = wave_force[2]
plt.figure()
plt.plot(t, Fx, label='Surge Force Fx [N]')
plt.plot(t, Fy, label='Sway Force Fy [N]')
# plt.plot(t, M, label='Moment [Nm]')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.title('Wave-Induced Forces (Froude-Krylov Only)')
plt.grid(True)
plt.show()