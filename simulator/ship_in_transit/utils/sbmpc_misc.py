import numpy as np, math

def wrap_min_max(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """Wraps input x to [x_min, x_max)

    Args:
        x (float or np.ndarray): Unwrapped value
        x_min (float or np.ndarray): Minimum value
        x_max (float or np.ndarray): Maximum value

    Returns:
        float or np.ndarray: Wrapped value
    """
    if isinstance(x, np.ndarray):
        return x_min + np.mod(x - x_min, x_max - x_min)
    else:
        return x_min + (x - x_min) % (x_max - x_min)


def wrap_angle_to_pmpi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps input angle to [-pi, pi)

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    if isinstance(angle, np.ndarray):
        return wrap_min_max(angle, -np.pi * np.ones(angle.size), np.pi * np.ones(angle.size))
    else:
        return wrap_min_max(angle, -np.pi, np.pi)
    
class Obstacle:
    def __init__(self, state: np.ndarray, T: np.double, dt: np.double):
        self.n_samp_ = int(T / dt)

        self.T_ = T
        self.dt_ = dt

        self.x_ = np.zeros(self.n_samp_)
        self.y_ = np.zeros(self.n_samp_)
        self.u_ = np.zeros(self.n_samp_)
        self.v_ = np.zeros(self.n_samp_)


        self.x_[0] = state[1][0]
        self.y_[0] = state[1][1]
        # V_x = state[1][2]
        # V_y = state[1][3]
        # self.psi_ = np.arctan2(V_y, V_x) - math.pi/2 # chi
        self.psi_ = state[1][2]

        self.l = state[3]
        self.w = state[4]

        # self.r11_ = np.cos(self.psi_)
        # self.r12_ = -np.sin(self.psi_)
        # self.r21_ = np.sin(self.psi_)
        # self.r22_ = np.cos(self.psi_)

        # self.u_[0] = self.r22_ * V_x + self.r21_ * V_y
        # self.v_[0] = self.r12_ * V_x + self.r11_ * V_y

        self.r11_ = -np.sin(self.psi_) # --> Rotation matrix to bring uvr into world, using the desired heading
        self.r12_ = np.cos(self.psi_)
        self.r21_ = np.cos(self.psi_)
        self.r22_ = np.sin(self.psi_)

        # u = Vy * cos(psi) - Vx * sin(psi)
        # v = Vy * sin(psi) + Vx * cos(psi)

        self.u_[0] = state[1][3] # self.r11_ * V_x + self.r12_ * V_y
        self.v_[0] = state[1][4] # self.r21_ * V_x + self.r22_ * V_y

        self.calculate_trajectory()

    def calculate_trajectory(self):
        for i in range(1, self.n_samp_):
            self.x_[i] = self.x_[i - 1] + (self.r11_ * self.u_[i - 1] + self.r12_ * self.v_[i - 1]) * self.dt_
            self.y_[i] = self.y_[i - 1] + (self.r21_ * self.u_[i - 1] + self.r22_ * self.v_[i - 1]) * self.dt_
            self.u_[i] = self.u_[i - 1]
            self.v_[i] = self.v_[i - 1]
    
class ShipLinearModel:
    def __init__(self, T: np.double, dt: np.double, length:float=25, width:float=80):
        self.n_samp_ = int(T / dt)

        self.T_ = T
        self.DT_ = dt

        self.x_ = np.zeros(self.n_samp_)
        self.y_ = np.zeros(self.n_samp_)
        self.psi_ = np.zeros(self.n_samp_)
        self.u_ = np.zeros(self.n_samp_)
        self.v_ = np.zeros(self.n_samp_)
        self.r_ = np.zeros(self.n_samp_)

        self.l = length
        self.w = width


    def linear_pred(self, state, u_d, psi_d):
        self.psi_[0] = wrap_angle_to_pmpi(psi_d)

        self.x_[0] = state[0]
        self.y_[0] = state[1]
        self.u_[0] = u_d
        self.v_[0] = state[4]
        self.r_[0] = 0

        r11 = -np.sin(psi_d) # --> Rotation matrix to bring uvr into world, using the desired heading
        r12 = np.cos(psi_d)
        r21 = np.cos(psi_d)
        r22 = np.sin(psi_d)

        for i in range(1, self.n_samp_):
            self.x_[i] = self.x_[i - 1] + self.DT_ * (r11 * self.u_[i - 1] + r12 * self.v_[i - 1]) # Output is in world frame
            self.y_[i] = self.y_[i - 1] + self.DT_ * (r21 * self.u_[i - 1] + r22 * self.v_[i - 1])
            self.psi_[i] = psi_d  # self.psi_[i-1] + self.DT_*self.r_[i-1]
            self.u_[i] = u_d  # self.u_[i-1] + self.DT_*(u_d-self.u_[i-1])
            self.v_[i] = 0
            self.r_[i] = 0  # math.atan2(np.sin(psi_d - self.psi_[i-1]), np.cos(psi_d - self.psi_[i-1]))