from pathlib import Path
import sys

from simulator.ship_in_transit.sub_systems.env_load_prob_model import SeaStateMixture

# PATH HELPER
# project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Example usage:
sampler = SeaStateMixture()
draw = sampler.sample_joint()
print(draw["Hs"], draw["Uw"], draw["Tp"], draw["logp_total"])
logp_marg = sampler.logpdf_marginal(draw["Hs"], draw["Uw"], draw["Tp"])
print("marginal logp:", logp_marg)
