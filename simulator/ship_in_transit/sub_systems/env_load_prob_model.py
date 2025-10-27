import math
import numpy as np
from scipy.stats import truncnorm, norm, vonmises
from scipy.special import logsumexp

# ---------- log-pdfs for base distributions ----------
def logpdf_uniform(x, a, b):
    if a <= x <= b and b > a:
        return -math.log(b - a)
    return -np.inf

def logpdf_triangular(x, a, m, b):
    # a <= m <= b; piecewise-linear pdf
    if not (a <= x <= b) or not (a <= m <= b) or b == a:
        return -np.inf
    if x == m:
        # handle peak (use side limit)
        left = 0.0 if m == a else 2.0 / (b - a)
        right = 0.0 if m == b else 2.0 / (b - a)
        val = max(left, right)
        return -np.inf if val <= 0 else math.log(val)
    if x < m:
        num = 2*(x - a)
        den = (b - a)*(m - a)
    else:
        num = 2*(b - x)
        den = (b - a)*(b - m)
    if num <= 0 or den <= 0:
        return -np.inf
    return math.log(num) - math.log(den)

def make_truncnorm(a, m, b, sigma_frac=0.25):
    width = b - a
    sigma = max(1e-9, sigma_frac * width)
    low, high = (a - m) / sigma, (b - m) / sigma
    return truncnorm(low, high, loc=m, scale=sigma)

def logpdf_truncnorm(x, a, m, b, sigma_frac=0.25):
    return float(make_truncnorm(a, m, b, sigma_frac).logpdf(x))

def rvs_param(spec):
    """
    * Random Variate Sampling
    Sample x from the parameter distribution and get its log-probability under the same the same distribution.
    """
    a, b = spec["range"]
    m = spec["mean"]
    dist = spec.get("dist", "triangular")
    if dist == "uniform":
        x = float(np.random.uniform(a, b))
        ll = logpdf_uniform(x, a, b)
    elif dist == "truncnorm":
        tn = make_truncnorm(a, m, b, spec.get("sigma_frac", 0.25))
        x = float(tn.rvs())
        ll = float(tn.logpdf(x))
    else:
        x = float(np.random.triangular(a, m, b))
        ll = logpdf_triangular(x, a, m, b)
    return x, ll

def logpdf_param(x, spec):
    a, b = spec["range"]
    m = spec["mean"]
    dist = spec.get("dist", "triangular")
    if dist == "uniform":
        return logpdf_uniform(x, a, b)
    elif dist == "truncnorm":
        return logpdf_truncnorm(x, a, m, b, spec.get("sigma_frac", 0.25))
    else:
        return logpdf_triangular(x, a, m, b)
    
def wrap_angle(x):
    # wrap to (-pi, pi]
    return (x + np.pi) % (2*np.pi) - np.pi

# ----- Prior over current-speed MEAN (independent of S) -----
def logprior_mu_speed(mu_c, range=(0.0, 2.5), center=0.7, sigma_frac=0.25):
    """
    Broad truncated-normal prior for current mean speed (m/s).
    Tweak rng/center/sigma_frac to your climatology.
    
    center is the previously sampled mean current from the RL-agent
    mu_c is the newly sampled mean current from the RL-agent
    """
    a, b = range
    return logpdf_truncnorm(mu_c, a, center, b, sigma_frac=sigma_frac)

# ----- Prior over MEAN directions (wind/wave/current) -----
def logprior_mu_direction(mu_dir, clim_mean_dir, kappa0=3.0):
    """
    Von Mises prior for the *mean direction* around some climatological mean.
    If you want 'uninformative', set kappa0 ~ 0 (uniform on circle).
    
    clim_mean_dir is the previously sampled mean direction from the RL-agent
    mu_dir is the newly sampled mean direction from the RL-agent.
    """
    # We score at the exact mean direction value. Von Mises is continuous,
    # so using the pdf at a point is fine in log-domain.
    return float(vonmises.logpdf(wrap_angle(mu_dir), kappa0, loc=wrap_angle(clim_mean_dir)))

# ---------- Probabilities Mixture over sea states ----------
# --------------------- Hs, Tp, Uw --------------------------
class SeaStateMixture:
    """
    states: list of dicts like:
      { "name": str, "p": float,
        "Hs": {"range": (a,b), "mean": m, "dist": "triangular|uniform|truncnorm", ...},
        "Uw": {...}, "Tp": {...} }
    Assumes conditional independence of Hs, Uw, Tp given S.
    """
    def __init__(self):
        # ---------- North Atlantic states (Lee et al., 1985) ----------
        # Dist choices:
        # - Tp: triangular with mode = "most probable" from the table
        # - Hs, Uw: truncated normal centered at the table mean, bounded by the range
        # - sigma_frac set to 0.25, default of the range width
        self.states = [
            {"name":"SS 0–1","p":0.0070,
            "Hs":{"range":(0.0,0.1),"mean":0.05,"dist":"truncnorm","sigma_frac":0.25},
            "Uw":{"range":self.knot_to_ms([0.0,6.0]),"mean":self.knot_to_ms(3.0),"dist":"truncnorm","sigma_frac":0.25},
            "Tp":{"range":(0.0,3.3),"mean":1.65,"dist":"triangular"}},   # Tp placeholder
            {"name":"SS 2","p":0.0680,
            "Hs":{"range":(0.1,0.5),"mean":0.3,"dist":"truncnorm","sigma_frac":0.25},
            "Uw":{"range":self.knot_to_ms([7.0,10.0]),"mean":self.knot_to_ms(8.5),"dist":"truncnorm","sigma_frac":0.25},
            "Tp":{"range":(3.3,12.8),"mean":7.5,"dist":"triangular"}},
            {"name":"SS 3","p":0.2370,
            "Hs":{"range":(0.5,1.25),"mean":0.88,"dist":"truncnorm","sigma_frac":0.25},
            "Uw":{"range":self.knot_to_ms([11.0,16.0]),"mean":self.knot_to_ms(13.5),"dist":"truncnorm","sigma_frac":0.25},
            "Tp":{"range":(5.0,14.8),"mean":7.5,"dist":"triangular"}},
            {"name":"SS 4","p":0.2780,
            "Hs":{"range":(1.25,2.5),"mean":1.88,"dist":"truncnorm","sigma_frac":0.25},
            "Uw":{"range":self.knot_to_ms([17.0,21.0]),"mean":self.knot_to_ms(19.0),"dist":"truncnorm","sigma_frac":0.25},
            "Tp":{"range":(6.1,15.2),"mean":8.8,"dist":"triangular"}},
            {"name":"SS 5","p":0.2064,
            "Hs":{"range":(2.5,4.0),"mean":3.25,"dist":"truncnorm","sigma_frac":0.25},
            "Uw":{"range":self.knot_to_ms([22.0,27.0]),"mean":self.knot_to_ms(24.5),"dist":"truncnorm","sigma_frac":0.25},
            "Tp":{"range":(8.3,15.5),"mean":9.7,"dist":"triangular"}},
            {"name":"SS 6","p":0.1315,
            "Hs":{"range":(4.0,6.0),"mean":5.0,"dist":"truncnorm","sigma_frac":0.25},
            "Uw":{"range":self.knot_to_ms([28.0,47.0]),"mean":self.knot_to_ms(37.5),"dist":"truncnorm","sigma_frac":0.25},
            "Tp":{"range":(9.8,16.2),"mean":12.4,"dist":"triangular"}},
            {"name":"SS 7","p":0.0605,
            "Hs":{"range":(6.0,9.0),"mean":7.5,"dist":"truncnorm","sigma_frac":0.25},
            "Uw":{"range":self.knot_to_ms([48.0,55.0]),"mean":self.knot_to_ms(51.5),"dist":"truncnorm","sigma_frac":0.25},
            "Tp":{"range":(11.8,18.5),"mean":15.0,"dist":"triangular"}},
            {"name":"SS 8","p":0.0111,
            "Hs":{"range":(9.0,14.0),"mean":11.5,"dist":"truncnorm","sigma_frac":0.25},
            "Uw":{"range":self.knot_to_ms([56.0,63.0]),"mean":self.knot_to_ms(59.5),"dist":"truncnorm","sigma_frac":0.25},
            "Tp":{"range":(14.2,18.6),"mean":16.4,"dist":"triangular"}},
            {"name":"SS >8","p":0.0005,
            "Hs":{"range":(14.0,20.0),"mean":16.0,"dist":"truncnorm","sigma_frac":0.25},  # capped
            "Uw":{"range":self.knot_to_ms([63.0,80.0]),"mean":self.knot_to_ms(70.0),"dist":"truncnorm","sigma_frac":0.25},  # capped
            "Tp":{"range":(18.0,23.7),"mean":20.0,"dist":"triangular"}},
        ]
        ps = np.array([s["p"] for s in self.states], dtype=float)
        ps = ps / ps.sum()
        self.log_ps = np.log(ps)
        
    # ---- Utilities ---- #
    def knot_to_ms(self, knot):
        if isinstance(knot, (list, tuple, np.ndarray)):
            return [k * 0.51444444 for k in knot]
        else:
            return knot * 0.51444444

    def ms_to_knot(self, ms):
        if isinstance(ms, (list, tuple, np.ndarray)):
            return [m / 0.51444444 for m in ms]
        else:
            return ms / 0.51444444

    # ---- sampling + joint log-prob (when S is sampled) ----
    def sample_joint(self):
        '''
        ll = log-likelihood
        '''
        idx = int(np.random.choice(len(self.states), p=np.exp(self.log_ps)))
        s = self.states[idx]
        hs, ll_hs = rvs_param(s["Hs"])
        uw, ll_uw = rvs_param(s["Uw"])
        tp, ll_tp = rvs_param(s["Tp"]) 
        ll_state = float(self.log_ps[idx])
        return {
            "state_index": idx,
            "state_name": s["name"],
            "Hs": hs, "Uw": uw, "Tp": tp,
            "logp_state": ll_state,
            "logp_params": ll_hs + ll_uw + ll_tp,
            "logp_total": ll_state + ll_hs + ll_uw + ll_tp
        }

    # ---- conditional log-pdf given a specific state ----
    def logpdf_given_state(self, hs, uw, tp, idx):
        s = self.states[idx]
        return (
            logpdf_param(hs, s["Hs"]) +
            logpdf_param(uw, s["Uw"]) +
            logpdf_param(tp, s["Tp"])
        )

    # ---- joint log-pdf of (S, hs, uw, tp) ----
    def logpdf_joint(self, hs, uw, tp, idx):
        return float(self.log_ps[idx] + self.logpdf_given_state(hs, uw, tp, idx))

    # ---- **marginal** log-pdf with S unknown: log sum_S p(S) f(.|S) ----
    def logpdf_marginal(self, hs, uw, tp):
        logs = []
        for i, _ in enumerate(self.states):
            logs.append(self.log_ps[i] + self.logpdf_given_state(hs, uw, tp, i))
        return float(logsumexp(logs))