from numba import int64
from numba.experimental import jitclass
import numpy as np


@jitclass
class Logs:
    """
    Jitclass for logs.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    m: :class:`int`
        Number of edges.
    max_queue: :class:`int`
        Maximum number of items per class.
    """
    income: int64[:]
    traffic: int64[:]
    queue_log: int64[:, :]

    def __init__(self, n, m, max_queue):
        self.income = np.zeros(n, dtype=np.int64)
        self.traffic = np.zeros(m, dtype=np.int64)
        self.queue_log = np.zeros((n, max_queue), dtype=np.int64)

    @property
    def steps_done(self):
        return np.sum(self.income)

    def update(self, queue_size, node, edge):
        self.income[node] += 1
        if edge > -1:
            self.traffic[edge] += 1
        for i, q in enumerate(queue_size):
            self.queue_log[i, q] += 1


@jitclass
class PhantomLogs:
    """
    Jitclass for logs of e-filtering policies. automatically concatenates the results in the original format.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes of the original graph.
    m: :class:`int`
        Number of edges of the original graph.
    max_queue: :class:`int`
        Maximum number of items per class.
    edges: :class:`~numpy.ndarray`
        Edge index.
    """
    income: int64[:]
    traffic: int64[:]
    edges: int64[:]
    queue_log: int64[:, :]
    steps_done: int
    n: int

    def __init__(self, n, m, max_queue, edges):
        self.income = np.zeros(n, dtype=np.int64)
        self.traffic = np.zeros(m, dtype=np.int64)
        self.edges = edges
        self.queue_log = np.zeros((n, max_queue), dtype=np.int64)
        self.steps_done = 0
        self.n = n

    def update(self, queue_size, node, edge):
        self.income[node % self.n] += 1
        if edge > -1:
            self.traffic[self.edges[edge]] += 1
        for i in range(self.n):
            q = queue_size[i] + queue_size[i+self.n]
            self.queue_log[i, q] += 1
        self.steps_done += 1


def repr_logs(logs):
    """
    Parameters
    ----------
    logs: :class:`~stochastic_matching.simulator.logs.Logs`
        Logs to display. Relies on `print`.

    Returns
    -------
    None
    """
    print(f"Arrivals: {logs.income.astype(int)}\n"
          f"Traffic: {logs.traffic.astype(int)}\n"
          f"Queues: {logs.queue_log.astype(int)}\n"
          f"Steps done: {logs.steps_done}")
