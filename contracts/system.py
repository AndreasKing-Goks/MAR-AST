# ---------------------------------------------------------------------------
# 7. System-level Contract
# ---------------------------------------------------------------------------

class SystemLevelContract:
    NAME = "SystemLevel"
    MESSAGES = {
        "A1": "Route is outside chart horizon or intersects static obstacles.",
        "A2": "Simulation time exceeds specified horizon.",
        "G1": "Grounding event occurred.",
        "G2": "Ship left scenario horizon.",
        "G3": "Navigation success criteria not satisfied.",
        "G4": "Propulsion overload events occurred.",
        "G5": "Travel distance or time out of expected bounds.",
    }


    def __init__(
        self,
        route_inside_chart: bool,
        free_of_static_obstacles: bool,
        sim_time: float,
        sim_time_max: float,
        grounding_events: int,
        left_scenario_horizon: bool,
        navigation_success: bool,
        propulsion_overload_events: int,
        travel_distance: float,
        travel_time: float,
        distance_min: float,
        distance_max: float,
        time_min: float,
        time_max: float,
    ):
        self.route_inside_chart = route_inside_chart
        self.free_of_static_obstacles = free_of_static_obstacles
        self.sim_time = sim_time
        self.sim_time_max = sim_time_max
        self.grounding_events = grounding_events
        self.left_scenario_horizon = left_scenario_horizon
        self.navigation_success = navigation_success
        self.propulsion_overload_events = propulsion_overload_events
        self.travel_distance = travel_distance
        self.travel_time = travel_time
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.time_min = time_min
        self.time_max = time_max

        self.contract_status = {
            "A1": None, "A2": None,
            "G1": None, "G2": None, "G3": None, "G4": None, "G5": None
        }

    def check_A1(self):
        self.contract_status["A1"] = bool(self.route_inside_chart and self.free_of_static_obstacles)
        return self.contract_status["A1"]

    def check_A2(self):
        self.contract_status["A2"] = bool(self.sim_time <= self.sim_time_max)
        return self.contract_status["A2"]

    def check_G1(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G1"] = None
        #     return None
        self.contract_status["G1"] = self.grounding_events == 0
        return self.contract_status["G1"]

    def check_G2(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G2"] = None
        #     return None
        self.contract_status["G2"] = not self.left_scenario_horizon
        return self.contract_status["G2"]

    def check_G3(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G3"] = None
        #     return None
        self.contract_status["G3"] = bool(self.navigation_success)
        return self.contract_status["G3"]

    def check_G4(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G4"] = None
        #     return None
        self.contract_status["G4"] = self.propulsion_overload_events == 0
        return self.contract_status["G4"]

    def check_G5(self):
        # if not all(self.contract_status[a] for a in ["A1", "A2"]):
        #     self.contract_status["G5"] = None
        #     return None
        d_ok = bool(self.distance_min <= self.travel_distance <= self.distance_max)
        t_ok =bool(self.time_min <= self.travel_time <= self.time_max)
        self.contract_status["G5"] = d_ok and t_ok
        return self.contract_status["G5"]

    def evaluate(self, logger=None, t=None, meta=None):
        self.check_A1()
        self.check_A2()
        self.check_G1()
        self.check_G2()
        self.check_G3()
        self.check_G4()
        self.check_G5()

        if logger is not None and t is not None:
            for clause, value in self.contract_status.items():
                if value is False:
                    msg = self.MESSAGES.get(
                        clause, f"{self.NAME} contract {clause} violated."
                    )
                    logger.log(t, self.NAME, clause, msg)

        return self.contract_status
