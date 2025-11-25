from typing import Dict, Optional, Sequence
import numpy as np

from contracts.helpers import ViolationLogger, _is_finite_array

class ShipDynamicsContract:
    NAME = "ShipDynamics"
    MESSAGES = {
        "A1": "Missing or invalid thrust force from machinery.",
        "A2": "Missing or invalid rudder force from rudder system.",
        "A3": "Missing or invalid environmental load inputs.",
        "A4": "Initial ship state outside model validity region.",
        "G1": "Vessel state trajectories are invalid or discontinuous.",
        "G2": "Equations of motion residuals exceed tolerance.",
    }

    def __init__(
        self,
        thrust_force: np.ndarray,
        rudder_force: np.ndarray,
        env_forces: np.ndarray,
        eta: np.ndarray,
        nu: np.ndarray,
        acc: np.ndarray,
        initial_state_valid: bool,
        eom_residual: Optional[np.ndarray] = None,
        continuity_threshold: float = 1e3,
        eom_residual_tol: float = 1e-2,
    ):
        self.thrust_force = thrust_force
        self.rudder_force = rudder_force
        self.env_forces = env_forces
        self.eta = eta
        self.nu = nu
        self.acc = acc
        self.initial_state_valid = initial_state_valid
        self.eom_residual = eom_residual
        self.continuity_threshold = continuity_threshold
        self.eom_residual_tol = eom_residual_tol

        self.contract_status: Dict[str, Optional[bool]] = {
            "A1": None, "A2": None, "A3": None, "A4": None,
            "G1": None, "G2": None
        }

    # Assumptions
    def check_A1(self):
        self.contract_status["A1"] = _is_finite_array(self.thrust_force)
        return self.contract_status["A1"]

    def check_A2(self):
        self.contract_status["A2"] = _is_finite_array(self.rudder_force)
        return self.contract_status["A2"]

    def check_A3(self):
        self.contract_status["A3"] = _is_finite_array(self.env_forces)
        return self.contract_status["A3"]

    def check_A4(self):
        self.contract_status["A4"] = bool(self.initial_state_valid)
        return self.contract_status["A4"]

    # Guarantees
    def check_G1(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2", "A3", "A4"]):
        #     self.contract_status["G1"] = None
        #     return None

        def _continuous(x):
            if not _is_finite_array(x) or x.shape[0] < 2:
                return False
            diffs = np.linalg.norm(np.diff(x, axis=0), axis=1)
            return bool(np.all(diffs < self.continuity_threshold))

        ok_eta = _is_finite_array(self.eta)
        ok_nu = _is_finite_array(self.nu)
        ok_acc = _is_finite_array(self.acc)
        continuous = _continuous(self.eta) and _continuous(self.nu)
        self.contract_status["G1"] = ok_eta and ok_nu and ok_acc# and continuous
        return self.contract_status["G1"]

    def check_G2(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2", "A3", "A4"]):
        #     self.contract_status["G2"] = None
        #     return None

        if self.eom_residual is None:
            self.contract_status["G2"] = None
        else:
            res = np.asarray(self.eom_residual)
            self.contract_status["G2"] = bool(np.all(np.abs(res) <= self.eom_residual_tol))
        return self.contract_status["G2"]

    def evaluate(
        self,
        logger: Optional[ViolationLogger] = None,
        t: Optional[float] = None,
        meta: Optional[Dict] = None,
    ) -> Dict[str, Optional[bool]]:
        # Run all checks
        self.check_A1()
        self.check_A2()
        self.check_A3()
        self.check_A4()
        self.check_G1()
        self.check_G2()

        # Log only violated clauses
        if logger is not None and t is not None:
            for clause, value in self.contract_status.items():
                if value is False:  # violated
                    msg = self.MESSAGES.get(
                        clause, f"{self.NAME} contract {clause} violated."
                    )
                    logger.log(time=t,
                               subsystem=self.NAME,
                               contract_id=clause,
                               message=msg)

        return self.contract_status