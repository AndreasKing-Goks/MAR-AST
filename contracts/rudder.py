# ---------------------------------------------------------------------------
# 5. Rudder Contract
# ---------------------------------------------------------------------------

from typing import Dict, Optional, Sequence
import numpy as np
import os
import csv
import json
from contracts.helpers import _is_finite_array, ViolationLogger

class RudderContract:
    NAME = "Rudder"
    MESSAGES = {
        "A1": "Missing or invalid desired rudder angle from heading controller.",
        "G1": "Rudder angle exceeds mechanical limits.",
        "G2": "Rudder angle rate exceeds actuator limits.",
    }

    def __init__(
        self,
        desired_angle: np.ndarray,
        actual_angle: np.ndarray,
        angle_limit: float,
        max_rate: float,
    ):
        self.desired_angle = desired_angle
        self.actual_angle = actual_angle
        self.angle_limit = angle_limit
        self.max_rate = max_rate
        self.contract_status = {"A1": None, "G1": None, "G2": None}

    def check_A1(self):
        self.contract_status["A1"] = _is_finite_array(self.desired_angle)
        return self.contract_status["A1"]

    def check_G1(self):
        # if not self.contract_status["A1"]:
        #     self.contract_status["G1"] = None
        #     return None
        a = np.asarray(self.actual_angle)
        self.contract_status["G1"] = bool(np.all(np.abs(a) <= self.angle_limit))
        return self.contract_status["G1"]

    def check_G2(self):
        # if not self.contract_status["A1"]:
        #     self.contract_status["G2"] = None
        #     return None
        a = np.asarray(self.actual_angle)
        if a.size < 2:
            self.contract_status["G2"] = True
        else:
            rate = np.abs(np.diff(a))
            self.contract_status["G2"] = bool(np.all(rate <= self.max_rate))
        return self.contract_status["G2"]

    def evaluate(
        self,
        logger: Optional[ViolationLogger] = None,
        t: Optional[float] = None,
        meta: Optional[Dict] = None,
    ):
        self.check_A1()
        self.check_G1()
        self.check_G2()

        if logger is not None and t is not None:
            for clause, value in self.contract_status.items():
                if value is False:
                    msg = self.MESSAGES.get(
                        clause, f"{self.NAME} contract {clause} violated."
                    )
                    logger.log(time=t,
                               subsystem=self.NAME,
                               contract_id=clause,
                               message=msg)

        return self.contract_status