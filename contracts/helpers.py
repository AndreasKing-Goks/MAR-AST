import numpy as np
import pandas as pd
import os
import csv

from typing import Dict, Optional, Sequence

# ---------------------------------------------------------------------------
# Violation logger
# ---------------------------------------------------------------------------
class ViolationLogger:
    """
    Log contract violations in a human-friendly format:

    time, subsystem, contract_id, message

    Example:
    0.0,THRUST,A2,One or more thrusters are not working.
    """

    def __init__(self, csv_path: str, append: bool = False):
        self.csv_path = csv_path
        self.append = append
        self._ensure_header()

    def _ensure_header(self):
        file_exists = os.path.exists(self.csv_path)
        mode = "a" if (self.append and file_exists) else "w"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if not file_exists or not self.append:
                writer.writerow(["time", "subsystem", "contract_id", "message"])

    def log(self, time: float, subsystem: str, contract_id: str, message: str):
        """
        Log a single violated clause.
        """
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time, subsystem, contract_id, message])



# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_finite_array(x: Optional[np.ndarray]) -> bool:
    if x is None:
        return False
    x = np.asarray(x)
    return bool(x.size > 0 and np.all(np.isfinite(x)))