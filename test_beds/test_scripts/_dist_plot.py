import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises, truncnorm

from pathlib import Path
import sys

from simulator.ship_in_transit.sub_systems.env_load_prob_model import SeaStateMixture

# PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

test_1 = True
# test_1 = False

if test_1:
    # Example usage:
    sampler = SeaStateMixture()
    sampler.condition_by_max_state(max_state_name="SS 6")
    draw = sampler.sample_joint()
    print(draw["Hs"], draw["Uw"], draw["Tp"], draw["logp_total"])
    logp_marg = sampler.logpdf_marginal(draw["Hs"], draw["Uw"], draw["Tp"])
    print("marginal logp:", logp_marg)
    print("Action validity :", sampler.action_validity(draw["Hs"], draw["Uw"], draw["Tp"]))
    idx = sampler.matching_states(draw["Hs"], draw["Uw"], draw["Tp"])
    print("Sea states      :", sampler.states[idx[0]]["name"])

test_2 = True
test_2 = False

if test_2:
    loc = np.deg2rad(0)  # circular mean
    kappa = 0.5  # concentration
    
    sample_size = 1000
    sample = vonmises(loc=loc, kappa=kappa).rvs(sample_size)
    
    fig = plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection='polar')
    x = np.linspace(-np.pi, np.pi, 500)
    vonmises_pdf = vonmises.pdf(x, loc=loc, kappa=kappa)
    x_ticks = [-np.pi*2, -np.pi*3/2, -np.pi, -np.pi*1/2, 0, np.pi*1/2, np.pi, np.pi*3/2, np.pi*2 ]
    x_ticklabels = ['-2π', '-3π/2', '-π/2', '-π', 0, 'π', 'π/2', '3π/2', '2π']
    y_ticks = [0, 0.25, 0.5, 0.75, 1.00]
    
    left.plot(x, vonmises_pdf)
    left.set_xticks(x_ticks)
    left.set_xticklabels(x_ticklabels)
    left.set_yticks(y_ticks)
    number_of_bins = int(np.sqrt(sample_size))
    left.hist(sample, density=True, bins=number_of_bins)
    left.set_title("Cartesian plot")
    left.set_xlim(-np.pi, np.pi)
    left.set_ylim(0, 0.5)
    left.grid(True)
    
    right.plot(x, vonmises_pdf, label="PDF")
    right.set_yticks(y_ticks)
    right.set_rlim(0, 0.5)
    right.hist(sample, density=True, bins=number_of_bins,
            label="Histogram")
    right.set_title("Polar plot")
    right.grid(True)
    right.legend(bbox_to_anchor=(0.15, 1.06))
    right.set_theta_zero_location('N')   # 0° at North
    right.set_theta_direction(-1)        # CW positive
    right.set_thetagrids([0,45,90,135,180,225,270,315],
                      labels=['0°','45°','90°','135°','180°','-135°','-90°','-45°'])
    
    plt.show()
    
test_3 = True
test_3 = False

if test_3:
    fig, ax = plt.subplots(1, 1)
    a, b = 0.1, 2
    lb, ub = truncnorm.support(a, b)
    mean, var, skew, kurt = truncnorm.stats(a, b, moments='mvsk')
    x = np.linspace(truncnorm.ppf(0.01, a, b),
                truncnorm.ppf(0.99, a, b), 100)
    ax.plot(x, truncnorm.pdf(x, a, b),
        'r-', lw=5, alpha=0.6, label='truncnorm pdf')
    rv = truncnorm(a, b)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    vals = truncnorm.ppf([0.001, 0.5, 0.999], a, b)
    np.allclose([0.001, 0.5, 0.999], truncnorm.cdf(vals, a, b))
    r = truncnorm.rvs(a, b, size=1000)
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.set_xlim([x[0], x[-1]])
    ax.legend(loc='best', frameon=False)
    plt.show()
    
    ##############################################################################
    
    loc, scale = 1, 0.5
    rv = truncnorm(a, b, loc=loc, scale=scale)
    x = np.linspace(truncnorm.ppf(0.01, a, b),
                    truncnorm.ppf(0.99, a, b), 100)
    r = rv.rvs(size=1000)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.set_xlim(a, b)
    ax.legend(loc='best', frameon=False)
    plt.show()
    
    ##############################################################################
    
    a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
    rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
    x = np.linspace(truncnorm.ppf(0.01, a, b),
                    truncnorm.ppf(0.99, a, b), 100)
    r = rv.rvs(size=10000)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.set_xlim(a-0.1, b+0.1)
    ax.legend(loc='best', frameon=False)
    plt.show()