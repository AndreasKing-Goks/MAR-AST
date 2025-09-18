import numpy as np
from scipy.special import factorial

class JONSWAPWave:
    '''
    This class defines a wave model based on JONSWAP wave spectrum
    with wave spreading function.
    '''
    def __init__(self, s=1, w_min=0.4, w_max=2.5, N_omega=50, ship_length=80, ship_breadth=10, ship_draft=8,
                 psi_min=-np.pi/2, psi_max=np.pi/2, N_psi=10, seed=None, dt=30.0):
        '''
        Parameters:
        -----------
        omega_vec : float or np.array
            Angular frequencies [rad/s].
        psi_vec: float or np.array
            Discretized spreading angle [rad]
        k_vec : float or np.array
            Vector of wave numbers.
        gamma : float
            Peak enhancement factor (default ~3.3 for JONSWAP).
        g : float
            Gravity [m/s^2].
        s : int
            spreading parameter. 
                - s = 1 recommended by ITTC
                - s = 2 recommended by ISSC
        w_min: float
            The smallest available wave frequency.
        w_max: float
            The biggest available wave frequency.
        N_omega: int
            Discrete units count for the wave frequency discretization.
        psi_min: float
            The low bound for wave spread.
        psi_max: float
            The high bound for wave spread.
        N_psi: int
            Discrete units count for the wave spreading discretization.
        dt: float
            Time step size for time integration.
        '''
        
        # Self parameters
        self.g = 9.81
        self.s = s
        self.w_min = w_min
        self.w_max = w_max
        self.N_omega = N_omega
        self.psi_min = psi_min
        self.psi_max = psi_max
        self.N_psi = N_psi
        self.dt = dt
        self.ship_length = ship_length
        self.ship_breadth = ship_breadth
        self.ship_draft = ship_draft
        
        # Vector for each wave across all frequencies
        self.omega_vec = np.linspace(self.w_min, self.w_max, self.N_omega)
        self.domega = self.omega_vec[1] - self.omega_vec[0]
        
        # Vector for wave numbers
        self.k_vec = self.omega_vec**2 / self.g
        
        # Vector for each wave across discretized spreading direction
        self.psi_vec = np.linspace(-np.pi, np.pi, self.N_psi)
        self.dpsi = self.psi_vec[1] - self.psi_vec[0]
        
        # Random seed
        if seed is not None:
            np.random.seed(seed)
        
    def jonswap_spectrum(self, Hs, Tp, omega_vec):
        '''
        Returns S_eta(omega_vec): wave spectrum [m^2 s] across  all predetermined 
        wave frequencies using a JONSWAP formula in Faltinsen (1993).
        
        '''
        # Peak frequency
        wp = 2.0 * np.pi / Tp
        
        # Clip the wp based on Hs to keep within the validity area of the spectrum
        wp = np.clip(wp, 1.25/np.sqrt(Hs), 1.75/np.sqrt(Hs))
        
        # Alpha
        alpha = 0.2 * Hs**2 * wp**4 / self.g**2
        
        # Sigma (check across all frequency, then assign the sigma to each frequency)
        sigma = np.where(omega_vec <= wp, 0.07, 0.09)
        
        # Getting the gamma based on DNV GL [Sørensen (2018)]
        k = (2 * np.pi) / (wp * np.sqrt(Hs))
        
        if k <= 3.6:
            gamma = 5
        elif k <= 5.0:
            gamma = np.exp(5.75 - 1.15 * k)
        elif k > 5.0:
            gamma = 1
            
        # JONSWAP core
        gamma_exp = np.exp(- (omega_vec - wp)**2 / (2 * sigma**2 * wp**2))
        Sj = alpha * self.g**2 / omega_vec**5 * np.exp(-1.25*(wp/omega_vec)**4) * gamma**gamma_exp
        
        # Ensure non negative (negative energy density has no meaning at all)
        return np.maximum(Sj, 0.0)
    
    def spreading_function(self, psi_0, s, psi_vec):
        '''
        Returns D(psi): Spreading factor to spread the wave energy around the 
        mean wave direction
        '''
        delta_psi = psi_vec - psi_0
        
        spread_factor = (2**(2*s - 1) * factorial(s) * factorial(s-1)) / (np.pi * factorial(2*s - 1))

        
        # Core definition
        D = np.where(
            np.abs(delta_psi) < np.pi/2,
            spread_factor * np.cos(delta_psi)**(2*s),
            0.0
        )
        return D

    def get_wave_load(self, ship_speed, psi_ship, 
                          Hs, Tp, psi_0, 
                          omega_vec=None, psi_vec=None, s=None):
        '''
        Parameters:
        ship_speed: float
            Ship forward speed
        psi_ship: float
            Ship heading
        A_proj: flowat
            Projected ship section area at the waterline
        Hs : float
            Significant wave height [m].
        Tp : float
            Peak period [s].
        psi_0 : float
            Mean wave direction [rad]
            
        Returns (F_wx, F_wy): First order wave load computed with Froude-Krylof force approximation. [N]
        '''
        
        if omega_vec is None: omega_vec = self.omega_vec      # (Nw,)
        if psi_vec is None: psi_vec = self.psi_vec          # (Nd,)
        if s is None: s = self.s
        
        # Compute wave spectrum and the spreading function
        S_w = self.jonswap_spectrum(Hs, Tp, omega_vec)      # (Nw,)
        D_psi = self.spreading_function(psi_0, s, psi_vec)  # (Nd,)
        # Normalize the D to integrate to 1 over psi
        # So that the sum over D(psi) dpsi = 1
        D_psi = D_psi / (D_psi.sum() * self.dpsi)           # (Nd,)
        
        # Component elevation amplitudes
        a_eta = np.sqrt(2.0 * np.outer(S_w, D_psi) * self.domega * self.dpsi)   # (Nw, Nd)
        
        # Encounter correction [Forward speed effect in Faltinsen (1993)]
        beta = psi_vec[None, :] - psi_ship                                       # (1, Nd)
        omega_e = omega_vec[:, None] - self.k_vec[:, None] * ship_speed * np.cos(beta)    # (Nw, 1) - (Nw, 1)*(1, Nd) = (Nw, Nd)
        
        # Approximation of oblique wave
        beta_0 = psi_0 - psi_ship
        A_proj = (self.ship_breadth * np.cos(beta_0) + self.ship_length * np.sin(beta_0)) * self.ship_draft
        
        # Froude-Krylov flat-plate force amplitude
        rho = 1025.0
        F0 = rho * self.g * a_eta * A_proj
        
        # Direction unit vectors
        cx = np.cos(beta)  # (1, Nd)
        cy = np.sin(beta)  # (1, Nd)
        
        # Vector for randp, phases for each wave across all frequencies
        phases = 2.0 * np.pi * np.random.rand(self.N_omega, self.N_psi)    # (Nw, Nd)
        
        # Fx(t) = sum_{i,j} F0[i,j] * cos(omega_e[i,j]*t + phi[i,j]) * cos(theta_j)
        arg = omega_e * self.dt + phases
        cosarg = np.cos(arg)                            # (Nw, Nd)
        
        Fx_t = np.sum(F0 * cosarg * cx, axis=(0,1))     # (Sum across Nw and Nd)
        Fy_t = np.sum(F0 * cosarg * cy, axis=(0,1))     # (Sum across Nw and Nd)
        
        return Fx_t, Fy_t