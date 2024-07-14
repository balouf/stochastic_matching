import itertools
import numpy as np

from stochastic_matching.simulator.graph import make_jit_graph
from stochastic_matching.simulator.simulator import Simulator
from stochastic_matching.model import Model
from stochastic_matching.common import class_converter
from stochastic_matching.simulator.logs import PhantomLogs


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

    return Model(incidence=new_inc, rates=new_rates), np.array([e[0] for e in edge_codex]).astype(np.int64)


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
    **base_policy_kwargs:
        Keyword arguments. Only universal parameters should be used (seed, max_queue, n_steps).

    Examples
    --------

    Consider the following diamond graph with injective vertex [1, 2, 3]:

    >>> import stochastic_matching as sm
    >>> diamond = sm.CycleChain(rates=[1, 2, 2, 1])


    Without any specific argument, epsilon-filtering acts as a regular longest policy:

    >>> sim = EFiltering(diamond, n_steps=1000, seed=42)
    >>> sim.run()
    >>> sim.plogs
    Arrivals: [162 342 348 148]
    Traffic: [ 76  86 190  76  72]
    Queues: [[882  91  23 ...   0   0   0]
     [661 109  72 ...   0   0   0]
     [737 111  63 ...   0   0   0]
     [870  96  27 ...   0   0   0]]
    Steps done: 1000

    Let us use epsilon-filtering by specifying forbidden_edges:

    >>> sim = EFiltering(diamond, forbidden_edges=[0, 4], n_steps=1000, seed=42)
    >>> sim.run()
    >>> sim.plogs
    Arrivals: [162 342 348 148]
    Traffic: [  1 158 189 147   1]
    Queues: [[571  44  61 ...   0   0   0]
     [560  37  52 ...   0   0   0]
     [529  66  67 ...   0   0   0]
     [537  62  59 ...   0   0   0]]
    Steps done: 1000

    Switch to a FCFM policy:

    >>> sim = EFiltering(diamond, base_policy='fcfm', forbidden_edges=[0, 4], n_steps=1000, seed=42)
    >>> sim.run()
    >>> sim.plogs
    Arrivals: [162 342 348 148]
    Traffic: [  1 158 189 147   1]
    Queues: [[569  99 118 ...   0   0   0]
     [569  35  37 ...   0   0   0]
     [528  58  57 ...   0   0   0]
     [551  84  98 ...   0   0   0]]
    Steps done: 1000

    Switch to virtual queue:

    >>> sim = EFiltering(diamond, base_policy='virtual_queue', forbidden_edges=[0, 4], n_steps=1000, seed=42)
    >>> sim.run()
    >>> sim.plogs
    Arrivals: [162 342 348 148]
    Traffic: [  0 157 186 144   1]
    Queues: [[311  93 110 ...   0   0   0]
     [108 120 111 ...   0   0   0]
     [ 67 113  76 ...   0   0   0]
     [174 142 137 ...   0   0   0]]
    Steps done: 1000

    Stolyar's example to see the behavior on hypergraph:

    >>> stol = sm.Model(incidence=[[1, 0, 0, 0, 1, 0, 0],
    ...                  [0, 1, 0, 0, 1, 1, 1],
    ...                  [0, 0, 1, 0, 0, 1, 1],
    ...                  [0, 0, 0, 1, 0, 0, 1]], rates=[1.2, 1.5, 2, .8])
    >>> rewards = [-1, -1, 1, 2, 5, 4, 7]

    >>> sim = EFiltering(stol, base_policy='virtual_queue', rewards=rewards, n_steps=1000, epsilon=.0001, seed=42)
    >>> sim.run()
    >>> sim.plogs
    Arrivals: [216 282 371 131]
    Traffic: [  0   0 311  76 213   0  55]
    Queues: [[ 30 114 181 ...   0   0   0]
     [ 22  64 117 ...   0   0   0]
     [ 24 590 217 ...   0   0   0]
     [995   5   0 ...   0   0   0]]
    Steps done: 1000
    >>> sim.logs.traffic @ rewards
    1913
    >>> sim.avg_queues
    array([3.577, 4.856, 1.65 , 0.005])

    To compare with, the original EGPD policy:

    >>> sim = sm.VirtualQueue(stol, rewards=rewards, n_steps=1000, seed=42, beta=.01)
    >>> sim.run()
    >>> sim.plogs
    Arrivals: [236 279 342 143]
    Traffic: [  0   0 100   3 139   0 140]
    Queues: [[  2   3   5 ...   0   0   0]
     [844   8   6 ...   0   0   0]
     [  0   1   3 ...   0   0   0]
     [597 230  96 ...   0   0   0]]
    Steps done: 1000
    >>> sim.logs.traffic @ rewards
    1781
    >>> sim.avg_queues
    array([54.439,  1.597, 78.51 ,  0.691])
    """
    name = 'e_filtering'

    def __init__(self, model, base_policy='longest', forbidden_edges=None, rewards=None, epsilon=.01,
                 **base_policy_kwargs):
        if rewards is not None:
            rewards = np.array(rewards)
            flow = model.optimize_rates(rewards)
            forbidden_edges = [i for i in range(model.m) if flow[i] == 0]
            self.rewards = rewards
        else:
            self.rewards = np.ones(model.m, dtype=int)
            if forbidden_edges is None:
                forbidden_edges = []
            else:
                self.rewards[forbidden_edges] = -1
        self.base_policy = class_converter(base_policy, Simulator)
        self.base_policy_kwargs = base_policy_kwargs
        self.forbidden_edges = forbidden_edges
        self.epsilon = epsilon
        super().__init__(model, **base_policy_kwargs)

    def set_internal(self):
        expanded_model, edges = expand_model(model=self.model, forbidden_edges=self.forbidden_edges,
                                             epsilon=self.epsilon)
        expanded_simu = self.base_policy(model=expanded_model, **self.base_policy_kwargs)
        expanded_simu.logs = PhantomLogs(n=self.model.n, m=self.model.m, max_queue=self.max_queue, edges=edges)
        self.internal = {'simu': expanded_simu, 'n_steps': self.n_steps}

    def run(self):
        self.internal['simu'].run()
        self.logs = self.internal['simu'].logs
