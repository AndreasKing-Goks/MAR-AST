# ---------------------------------------------------------------------------
# 2. Environmental Loads Contract
# ---------------------------------------------------------------------------

from typing import Dict, Optional, Sequence
import numpy as np
import os
import csv
import json
from contracts.helpers import _is_finite_array

class EnvironmentalLoadsContract:
    NAME = "EnvironmentalLoads"
    MESSAGES = {
        "A1": "Missing environmental parameters (wind/current/etc.).",
        "A2": "Environmental parameters out of valid modelled range.",
        "G1": "Environmental loads are invalid (NaN/inf).",
        "G2": "Environmental forces exceed defined envelope.",
    }


    def __init__(
        self,
        params: Dict[str, float],
        param_ranges: Dict[str, Sequence[float]],
        forces: np.ndarray,
        force_envelope: Optional[Dict[str, float]] = None,
    ):
        self.params = params
        self.param_ranges = param_ranges
        self.forces = forces
        self.force_envelope = force_envelope or {}
        self.contract_status = {"A1": None, "A2": None, "G1": None, "G2": None}

    def check_A1(self):
        keys_ok = bool(all(k in self.params for k in self.param_ranges.keys()))
        vals_ok = bool(all(np.isfinite(self.params[k]) for k in self.param_ranges.keys()))
        self.contract_status["A1"] = keys_ok and vals_ok
        return self.contract_status["A1"]

    def check_A2(self):
        in_range = True
        for k, (lo, hi) in self.param_ranges.items():
            v = self.params.get(k, np.nan)
            if not (lo <= v <= hi):
                in_range = False
                break
        self.contract_status["A2"] = in_range
        return self.contract_status["A2"]

    def check_G1(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G1"] = None
        #     return None
        self.contract_status["G1"] = _is_finite_array(self.forces)
        return self.contract_status["G1"]

    def check_G2(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G2"] = None
        #     return None
        if "max_norm" in self.force_envelope:
            norms = np.linalg.norm(self.forces, axis=-1)
            self.contract_status["G2"] = bool(np.all(norms <= self.force_envelope["max_norm"]))
        else:
            self.contract_status["G2"] = None
        return self.contract_status["G2"]

    def evaluate(self, logger=None, t=None, meta=None):
        self.check_A1()
        self.check_A2()
        self.check_G1()
        self.check_G2()

        if logger is not None and t is not None:
            for clause, value in self.contract_status.items():
                if value is False:
                    msg = self.MESSAGES.get(
                        clause, f"{self.NAME} contract {clause} violated."
                    )
                    logger.log(t, self.NAME, clause, msg)

        return self.contract_status
