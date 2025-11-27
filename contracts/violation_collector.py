# multi_run_contract_logging.py

import pandas as pd

# Import your existing evaluation helper
# Adjust this import to match your actual module path:
# from contracts.contract_evaluation import evaluate_contracts_over_dataframe
from contracts.contracts import evaluate_contracts_over_dataframe



class InMemoryViolationLogger:
    def __init__(self):
        self.entries = []

    def log(self, time: float, subsystem: str, contract_id: str, message: str):
        self.entries.append(
            {
                "time": float(time),
                "subsystem": subsystem,
                "contract_id": contract_id,
                "message": message,
            }
        )

    def to_dataframe(self, run_id=None) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=["time", "subsystem", "contract_id", "message", "run_id"])
        df = pd.DataFrame(self.entries)
        if run_id is not None:
            df["run_id"] = run_id
        return df


class MultiRunContractLogger:
    """
    Aggregates contract violations over many simulation runs *in memory*.

    Usage pattern:

        mr_logger = MultiRunContractLogger()

        for run_id in range(runs):
            ... run env until termination ...
            df = pd.DataFrame.from_dict(env.assets[0].ship_model.simulation_results)
            mr_logger.run_once(df, env, run_id=run_id)

        all_violations = mr_logger.to_dataframe()
        pivot = mr_logger.summarize_by_contract()
        rates = mr_logger.violation_rate_by_contract()
    """

    def __init__(self):
        # List of per-run violation DataFrames
        self._run_dfs = []
        # Total timesteps across all runs
        self.total_timesteps = 0
        # Per-run lengths (for debugging / extra stats if needed)
        self.run_lengths = {}

    # ----------------------------------------------------------
    # Run contracts on ONE simulation and aggregate its violations
    # ----------------------------------------------------------
    def run_once(self, df: pd.DataFrame, env, run_id):
        """
        Evaluate contracts for a single run.

        Parameters
        ----------
        df : pd.DataFrame
            Simulation results for this run
            (from env.assets[0].ship_model.simulation_results).
        env : your environment object
            Passed through to evaluate_contracts_over_dataframe.
        run_id : any
            Identifier for this run (int, str, ...).
        """
        n_steps = len(df)
        self.total_timesteps += n_steps
        self.run_lengths[run_id] = n_steps

        # Use in-memory logger instead of CSV writer
        local_logger = InMemoryViolationLogger()

        # IMPORTANT: this uses your existing evaluation function
        # which calls each contract's .evaluate(logger=..., t=..., meta=...)
        evaluate_contracts_over_dataframe(df, env, logger=local_logger, run_id=run_id)

        # Convert this run's violations to DataFrame and tag with run_id
        run_df = local_logger.to_dataframe(run_id=run_id)
        self._run_dfs.append(run_df)

    # ----------------------------------------------------------
    # Aggregate all runs into a single DataFrame
    # ----------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame with all violations from all runs.

        Columns: time, subsystem, contract_id, message, run_id
        """
        if not self._run_dfs:
            return pd.DataFrame(
                columns=["time", "subsystem", "contract_id", "message", "run_id"]
            )
        return pd.concat(self._run_dfs, ignore_index=True)

    # ----------------------------------------------------------
    # Summaries (similar to your CSV-based summarize_violations_by_contract)
    # ----------------------------------------------------------
    def summarize_by_contract(self) -> pd.DataFrame:
        """
        Pivot table: rows=subsystem, columns=contract_id (A1,G1,...,TOTAL)
        showing counts of violations across ALL runs.
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        # Group by subsystem and specific contract_id (A1, G1, etc.)
        detailed_agg = (
            df.groupby(["subsystem", "contract_id"])
            .size()
            .reset_index(name="Violations")
        )

        # Add total per subsystem
        total_agg = df["subsystem"].value_counts().reset_index()
        total_agg.columns = ["subsystem", "Violations"]
        total_agg["contract_id"] = "TOTAL"

        # Combine detailed and totals
        combined = pd.concat([detailed_agg, total_agg], ignore_index=True)

        # Pivot for table format
        pivot = (
            combined.pivot(
                index="subsystem", columns="contract_id", values="Violations"
            )
            .fillna(0)
            .astype(int)
        )
        return pivot

    def violation_rate_by_contract(self) -> pd.Series:
        """
        Percentage of timesteps that had a violation of each contract_id,
        across ALL runs (based on total_timesteps).
        """
        df = self.to_dataframe()
        if df.empty or self.total_timesteps == 0:
            return pd.Series(dtype=float)

        counts = df.groupby("contract_id").size()
        rates = (counts / float(self.total_timesteps)) * 100.0
        rates.name = "violation_rate_percent"
        return rates.sort_values(ascending=False)

    def violation_rate_by_subsystem(self) -> pd.Series:
        """
        Percentage of timesteps that produced at least one violation per subsystem.
        """
        df = self.to_dataframe()
        if df.empty or self.total_timesteps == 0:
            return pd.Series(dtype=float)

        counts = df.groupby("subsystem").size()
        rates = (counts / float(self.total_timesteps)) * 100.0
        rates.name = "violation_rate_percent"
        return rates.sort_values(ascending=False)
