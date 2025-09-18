import numpy as np

class WindModel:
    """
    Mean wind:  Ȗ + μ Ȗ = w     (Gauss–Markov, Euler discretized)
    Gust:       U_g(t) = sum_i sqrt(2 S(f_i) Δf) cos(2π f_i t + φ_i)
    Direction:  ψ̇ + μψ ψ = wψ   (Gauss–Markov)
    """
    def __init__(self,
                 mu_Ubar, init_dir, mu_dir,
                 model="norsok",                 # "norsok" or "harris"
                 f_min=0.06, f_max=0.4, N_f=100,
                 init_Ubar=None, U10=2.5, z=5.0,
                 L=1800.0, kappa=0.0026,         # Harris params
                 sigma_Ubar=0.2, sigma_dir=0.05,
                 U_min=0.0, U_max=100.0,
                 dt=30.0, seed=None, clip_speed_nonnegative=True):

        # Random seed
        if seed is not None:
            np.random.seed(seed)

        # stochastic/mean params
        self.mu_Ubar, self.mu_dir = mu_Ubar, mu_dir
        self.sigma_Ubar, self.sigma_dir = sigma_Ubar, sigma_dir
        self.U_min, self.U_max = U_min, U_max
        self.dt = dt

        # environment / spectrum params
        self.model = model.lower()
        self.U10, self.z = U10, z      # common
        self.L, self.kappa = L, kappa  # Harris

        # frequency grid
        self.f = np.linspace(f_min, f_max, N_f)   # Hz
        self.df = self.f[1] - self.f[0]

        # mean wind at height z (log law used in Fossen around Harris example)
        if init_Ubar is None:
            z0 = 10.0 * np.exp(-2.0/(5.0*np.sqrt(self.kappa)))
            self.Ubar = self.U10 * (2.5*np.sqrt(self.kappa)) * np.log(self.z / z0)
        else:
            self.Ubar = float(init_Ubar)

        # direction state
        self.dir = float(init_dir)

        # ---- precompute gust amplitudes & phases (fixed), keep running angles ----
        S = self._spectrum(self.f)
        self.a = np.sqrt(2.0 * S * self.df)                  # (N_f,)
        self.phi0 = 2*np.pi*np.random.rand(N_f)              # fixed phases
        self.theta = self.phi0.copy()                        # running angles

        self.clip_speed_nonnegative = clip_speed_nonnegative

    # ----- spectra -----
    def _harris(self, f):
        # S(f) = 4 κ L U10 / (2 + (L f / U10)^2)^(5/6)
        ft = self.L * f / self.U10
        return 4.0 * self.kappa * self.L * self.U10 / (2.0 + ft**2)**(5.0/6.0)

    def _norsok(self, f):
        # S(f) = 320 (U10/10)^2 (z/10)^0.45 / (1 + x^n)^(5/(3n))
        # x = 172 f (z/10)^(2/3) (U10/10)^(-3/4), n = 0.468
        n = 0.468
        x = 172.0 * f * (self.z/10.0)**(2.0/3.0) * (self.U10/10.0)**(-3.0/4.0)
        return 320.0 * (self.U10/10.0)**2 * (self.z/10.0)**0.45 / (1.0 + x**n)**(5.0/(3.0*n))

    def _spectrum(self, f):
        if self.model == "harris":
            return self._harris(f)
        elif self.model == "norsok":
            return self._norsok(f)
        else:
            raise ValueError("model must be 'harris' or 'norsok'")

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