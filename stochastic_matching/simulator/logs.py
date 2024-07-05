from dataclasses import dataclass, fields
import numpy as np


@dataclass(repr=False)
class Logs:
    """
    A dataclass that facilitates the display of logs.

    Parameters
    ----------
    simu: :class:`~stochastic_matching.simulator.simulator.Simulator`
        Simulation for which logs are needed.
    """
    traffic: np.ndarray
    queue_log: np.ndarray
    steps_done: int

    def __init__(self, simu):
        self.traffic = np.zeros(simu.model.m, dtype=np.int64)
        self.queue_log = np.zeros((simu.model.n, simu.max_queue), dtype=np.int64)
        self.steps_done = 0

    def __repr__(self):
        return f"Traffic: {self.traffic}\nQueues: {self.queue_log}\nSteps done: {self.steps_done}"

    def asdict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}
