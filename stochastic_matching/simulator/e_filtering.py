import itertools
import numpy as np

from stochastic_matching.simulator.graph import make_jit_graph
from stochastic_matching.simulator.simulator import Simulator
from stochastic_matching.model import Model
from stochastic_matching.common import class_converter


def expand_model(model, forbidden_edges, epsilon):
    """
    Prepares a model for epsilon-filtering.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Initial model.
    forbidden_edges: :class:`list`
        "-/-" edges to remove in the expanded graph.
    epsilon: :class:`float`
        Probability to draw a "+" item.

    Returns
    -------
    model: :class:`~stochastic_matching.model.Model`
        Expanded model with 2n nodes that emulates epsilon-coloring.
    """
    n = model.n
    m = model.m
    graph = make_jit_graph(model)
    new_rates = np.concatenate([(1 - epsilon) * model.rates, epsilon * model.rates])
    edge_codex = list()
    for e in range(m):
        nodes = graph.nodes(e)
        for i, offset in enumerate(itertools.product([0, n], repeat=len(nodes))):
            if i == 0 and e in forbidden_edges:
                continue
            edge_codex.append((e, [v + o for v, o in zip(nodes, offset)]))
    new_inc = np.zeros((2 * n, len(edge_codex)), dtype=int)
    for i, nodes in enumerate(edge_codex):
        for node in nodes[1]:
            new_inc[node, i] = 1

    return Model(incidence=new_inc, rates=new_rates), [e[0] for e in edge_codex]


class EFiltering(Simulator):
    """
    Epsilon-filtering policy where incoming items are tagged with a spin.
    A match on a forbidden edge requires at least one "+" item.
    In practice, the simulator works on an expanded graph with twice the initial nodes to represent "+" and "-".

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    base_policy: :class:`str` or :class:`~stochastic_matching.simulator.simulator.Simulator`
        Type of simulator to instantiate. Cannot have mandatory extra-parameters (e.g. NOT 'priority').
    forbidden_edges: :class:`list` or :class:`~numpy.ndarray`, optional
        Edges that should not be used.
    weights: :class:`~numpy.ndarray`, optional
        Target rewards on edges. If weights are given, the forbidden edges are computed to match the target
        (overrides forbidden_edges argument).
    epsilon: :class:`float`, default=.01
        Proportion of "+" arrivals (lower means less odds to select a forbidden edge).
    **kwargs
        Keyword arguments.

    Notes
    -----

    Due to technical constraints, the queue_log does not contain the actual queue_log of the true model but the
    sum of the queue_log of the "+" and "-" items.
    For example, queue_log[0, 0] contains the sum of the times where 0+ is null and when 0- is null.
    This does not impact the computation of flows and average queues but the CCDF is not accurate
    and should not be used.

    Examples
    --------

    Consider the following diamond graph with injective vertex [1, 2, 3]:

    >>> import stochastic_matching as sm
    >>> diamond = sm.CycleChain(rates=[1, 2, 2, 1])

    Let us use epsilon-filtering.

    >>> sim = EFiltering(diamond, forbidden_edges=[0, 4], n_steps=1000, seed=42)
    >>> sim.run()
    >>> sim.logs
    Traffic: [  1 158 189 147   1]
    Queues: [[1572   44   61 ...    0    0    0]
     [1561   37   52 ...    0    0    0]
     [1529   66   67 ...    0    0    0]
     [1537   62   59 ...    0    0    0]]
    Steps done: 1000

    Notice the big queue_log. Each row sums to 2000 instead of 1000 because of state aggregation. It does not
    impact any computed metric apart from the CCDF, which is faulty.

    Switch to a FCFM policy:

    >>> sim = EFiltering(diamond, base_policy='fcfm', forbidden_edges=[0, 4], n_steps=1000, seed=42)
    >>> sim.run()
    >>> sim.logs
    Traffic: [  1 158 189 147   1]
    Queues: [[1570   99  118 ...    0    0    0]
     [1570   35   37 ...    0    0    0]
     [1528   58   57 ...    0    0    0]
     [1551   84   98 ...    0    0    0]]
    Steps done: 1000

    Switch to virtual queue:

    >>> sim = EFiltering(diamond, base_policy='virtual_queue', forbidden_edges=[0, 4], n_steps=1000, seed=42)
    >>> sim.run()
    >>> sim.logs
    Traffic: [  0 157 186 144   1]
    Queues: [[1312   93  110 ...    0    0    0]
     [1109  120  111 ...    0    0    0]
     [1068  113   76 ...    0    0    0]
     [1175  142  137 ...    0    0    0]]
    Steps done: 1000

    Stolyar's example to see the behavior on hypergraph:

    >>> stol = sm.Model(incidence=[[1, 0, 0, 0, 1, 0, 0],
    ...                  [0, 1, 0, 0, 1, 1, 1],
    ...                  [0, 0, 1, 0, 0, 1, 1],
    ...                  [0, 0, 0, 1, 0, 0, 1]], rates=[1.2, 1.5, 2, .8])
    >>> rewards = [-1, -1, 1, 2, 5, 4, 7]

    >>> sim = sm.EFiltering(stol, base_policy='virtual_queue', weights=rewards, n_steps=1000, epsilon=.0001, seed=42)
    >>> sim.run()
    >>> sim.logs
    Traffic: [  0   0 311  76 213   0  55]
    Queues: [[1031  114  181 ...    0    0    0]
     [1023   64  117 ...    0    0    0]
     [1025  590  217 ...    0    0    0]
     [1995    5    0 ...    0    0    0]]
    Steps done: 1000
    >>> sim.logs.traffic @ rewards
    1913
    >>> sim.compute_average_queues()
    array([3.574, 4.842, 1.645, 0.005])

    To compare with, the original EGPD policy:

    >>> sim = sm.VirtualQueue(stol, egpd_weights=rewards, n_steps=1000, seed=42)
    >>> sim.run()
    >>> sim.logs
    Traffic: [  0   0 100   3 139   0 140]
    Queues: [[  3   3   5 ...   0   0   0]
     [844   8   6 ...   0   0   0]
     [  1   1   3 ...   0   0   0]
     [597 230  96 ...   0   0   0]]
    Steps done: 1000
    >>> sim.logs.traffic @ rewards
    1781
    >>> sim.compute_average_queues()
    array([54.342,  1.597, 78.408,  0.691])
    """
    name = 'e_filtering'

    def __init__(self, model, base_policy='longest', forbidden_edges=None, weights=None, epsilon=.01,
                 **kwargs):
        if weights is not None:
            weights = np.array(weights)
            flow = model.optimize_rates(weights)
            forbidden_edges = [i for i in range(model.m) if flow[i] == 0]
        else:
            if forbidden_edges is None:
                forbidden_edges = []
        self.base_policy = class_converter(base_policy, Simulator)
        self.forbidden_edges = forbidden_edges
        self.epsilon = epsilon
        super().__init__(model, **kwargs)

    def set_internal(self):
        expanded_model, edges = expand_model(model=self.model, forbidden_edges=self.forbidden_edges,
                                                  epsilon=self.epsilon)
        expanded_simu = self.base_policy(model=expanded_model, n_steps=self.n_steps, max_queue=self.max_queue,
                                         seed=self.seed)
        self.internal = {'simu': expanded_simu, 'edges': edges, 'n_steps': self.n_steps}

    def run(self):
        n = self.model.n
        simu = self.internal['simu']
        edges = self.internal['edges']
        xlogs = simu.logs
        logs = self.logs
        simu.run()
        logs.steps_done = xlogs.steps_done
        logs.queue_log += (xlogs.queue_log[:n, :] + xlogs.queue_log[n:, :])
        for i, t in enumerate(xlogs.traffic):
            logs.traffic[edges[i]] += t
