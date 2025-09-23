import numpy as np
from numba import njit

from stochastic_matching.simulator.simulator import Simulator
from stochastic_matching.simulator.multiqueue import MultiQueue


@njit
def lin_fading(t):
    """

    Parameters
    ----------
    t: :class:`int`
        Step index.

    Returns
    -------
    :class:`int`
        Step index + 1
    """
    return t + 1


@njit
def cr_core(
    # Generic arguments
    logs,
    arrivals,
    graph,
    n_steps,
    queue_size,
    # Constant-regret specific arguments
    scores,
    ready_edges,
    edge_queue,
    fading
):
    """
    Jitted function for constant-regret policy.

    Parameters
    ----------
    logs: :class:`~stochastic_matching.simulator.logs.Logs`
        Monitored variables.
    arrivals: :class:`~stochastic_matching.simulator.arrivals.Arrivals`
        Item arrival process.
    graph: :class:`~stochastic_matching.simulator.graph.JitHyperGraph`
        Model graph.
    n_steps: :class:`int`
        Number of arrivals to process.
    queue_size: :class:`~numpy.ndarray`
        Number of waiting items of each class.
    scores: :class:`~numpy.ndarray`
        Normalized edge rewards (<0 on taboo edges). Enables EGPD-like selection mechanism.
    ready_edges: :class:`~numpy.ndarray`
        Tells if given edge is actionable.
    edge_queue: :class:`~stochastic_matching.simulator.multiqueue.MultiQueue`
        Edges waiting to be matched.
    fading: callable
        Jitted function that drives reward relative weight.

    Returns
    -------
    None
    """
    n, max_queue = logs.queue_log.shape
    m = len(scores)

    infinity = edge_queue.infinity

    edge_size = np.zeros(m, dtype=np.int32)

    for age in range(n_steps):
        # Draw an arrival
        node = arrivals.draw()
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return None

        # update scores
        edge_size[graph.edges(node)] += 1

        # update readyness
        if queue_size[node] == 1:
            for e in graph.edges(node):
                for v in graph.nodes(e):
                    if v != node and queue_size[v] == 0:
                        ready_edges[e] = False
                        break
                else:
                    ready_edges[e] = True

        # Select best edge
        best_edge = -1
        best_score = 0

        for e in range(m):
            score = scores[e] + edge_size[e] / fading(age)
            if score > best_score:
                best_edge = e
                best_score = score

        if best_edge > -1:
            # add edge to virtual queue
            edge_queue.add(best_edge, age)
            # Virtual pop of items:
            # for each node of the edge, lower scores of all adjacent edges by one
            for v in graph.nodes(best_edge):
                edge_size[graph.edges(v)] -= 1

        # Can a virtual edge be popped?
        best_edge = -1
        best_age = infinity
        for e in range(m):
            if ready_edges[e]:
                edge_age = edge_queue.oldest(e)
                if edge_age < best_age:
                    best_edge = e
                    best_age = edge_age
        if best_edge > -1:
            edge_queue.pop(best_edge)
            for v in graph.nodes(best_edge):
                queue_size[v] -= 1
                if queue_size[v] == 0:
                    ready_edges[graph.edges(v)] = False

        logs.update(queue_size=queue_size, node=node, edge=best_edge)
    return None


class ConstantRegret(Simulator):
    """
    Non-Greedy Matching simulator that implements https://sophieyu.me/constant_regret_matching.pdf
    Always pick-up the best positive edge according to a scoring function, even if that edge cannot be used (yet).

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    rewards: :class:`~numpy.ndarray`
        Edge rewards.
    fading: callable
        Jitted function that drives reward relative weight.
    max_edge_queue: :class:`int`, optional
        In some extreme situation, the default allocated space for the edge virtual queue may be too small.
        If that happens someday, use this parameter to increase the VQ allocated memory.
    **kwargs
        Keyword parameters of :class:`~stochastic_matching.simulator.simulator.Simulator`.


    Examples
    --------

    Let start with a working triangle.

    >>> import stochastic_matching as sm
    >>> sim = ConstantRegret(sm.Cycle(rates=[3, 4, 5]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [287 338 375]
    Traffic: [125 162 213]
    Queues: [[837 105  41  13   3   1   0   0   0   0]
     [784 131  53  22   8   2   0   0   0   0]
     [629 187  92  51  24   9   5   3   0   0]]
    Steps done: 1000

    Unstable diamond (simulation ends before completion due to drift).

    >>> sim = ConstantRegret(sm.CycleChain(rates='uniform'), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [85 82 85 86]
    Traffic: [34 42  7 41 36]
    Queues: [[141  55  34  25  23  16  24  12   7   1]
     [318  16   3   1   0   0   0   0   0   0]
     [316  17   4   1   0   0   0   0   0   0]
     [106  67  64  37  22  24   7   3   6   2]]
    Steps done: 338

    Stable diamond without reward optimization:

    >>> sim = ConstantRegret(sm.CycleChain(rates=[1, 2, 2, 1]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [179 342 320 159]
    Traffic: [ 95  84 161  84  75]
    Queues: [[823 120  38  19   0   0   0   0   0   0]
     [625 215  76  46  28   9   1   0   0   0]
     [686 212  70  24   7   1   0   0   0   0]
     [823 118  39  12   4   4   0   0   0   0]]
    Steps done: 1000

    Let's optimize (kill traffic on first and last edges).

    >>> sim = ConstantRegret(sm.CycleChain(rates=[1, 2, 2, 1]), rewards=[0, 1, 1, 1, 0],
    ...                    n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [20 44 51 23]
    Traffic: [ 3 17 25 16  0]
    Queues: [[136   2   0   0   0   0   0   0   0   0]
     [127   4   6   1   0   0   0   0   0   0]
     [ 24  22   9  13  14  20  16  13   4   3]
     [ 40  17  22  21  22  14   1   1   0   0]]
    Steps done: 138

    OK, it's mostly working, but we reached the maximal queue size quite fast. Let's reduce the pressure.

    >>> slow_fading = njit(lambda t: np.sqrt(t+1)/15)
    >>> sim = ConstantRegret(sm.CycleChain(rates=[1, 2, 2, 1]), rewards=[0, 1, 1, 1, 0],
    ...                    fading=slow_fading, n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [179 342 320 159]
    Traffic: [ 43 136 161 136  23]
    Queues: [[900  79  18   2   1   0   0   0   0   0]
     [840  98  37  17   6   2   0   0   0   0]
     [483 224 122  78  61  26   5   1   0   0]
     [551 255 100  50  25  10   9   0   0   0]]
    Steps done: 1000

    A stable candy. While candies are not good for greedy policies, virtual queues policies are
    designed to deal with it.

    >>> sim = ConstantRegret(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [140 126 154 112 224 121 123]
    Traffic: [109  29  17  59  58  62 107]
    Queues: [[301  83  94  83  54  60  43  48  41  32  49  60  31   3   8   7   3   0
        0   0   0   0   0   0   0]
     [825 101  27  14  26   7   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [899  64  20   8   9   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [110  49  93 108 120 102  48  28  53  32   7  30  16  42  28  34  31  30
       26  11   2   0   0   0   0]
     [276 185 151  95  60  40  49  52  47  33  12   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [755 142  67  36   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [709 228  47  13   3   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]]
    Steps done: 1000

    Last but not least: Stolyar's example from https://arxiv.org/abs/1608.01646

    >>> ns = sm.NS19(rates=[1.2, 1.5, 2, .8])

    Without optimization, all we do is self-matches (no queue at all):

    >>> sim = ConstantRegret(ns, n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs.traffic.astype(int)
    array([236, 279, 342, 143,   0,   0,   0])
    >>> sim.avg_queues
    array([0., 0., 0., 0.])

    Let's introduce some rewards:

    >>> rewards = [-1, -1, 1, 2, 5, 4, 7]
    >>> ns.optimize_rates(rewards)
    array([0. , 0. , 1.7, 0.5, 1.2, 0. , 0.3])
    >>> ns.normalize_rewards(rewards)
    array([-2., -5.,  0.,  0.,  0., -1.,  0.])

    With optimization, we get the desired results:

    >>> sim = ConstantRegret(ns, rewards=rewards, n_steps=3000, seed=42, max_queue=300)
    >>> sim.run()
    >>> sim.logs.traffic.astype(int)
    array([  0,   0, 961, 342, 691,   0, 105])
    >>> sim.avg_queues
    array([2.731     , 0.69633333, 0.22266667, 0.052     ])

    One can check that this is the same as using a regular virtual queue with forbidden edges:

    >>> from stochastic_matching.simulator.virtual_queue import VirtualQueue
    >>> sim = VirtualQueue(ns, rewards=rewards, forbidden_edges=True, n_steps=3000, seed=42, max_queue=300)
    >>> sim.run()
    >>> sim.logs.traffic.astype(int)
    array([  0,   0, 961, 342, 691,   0, 105])
    >>> sim.avg_queues
    array([2.731     , 0.69633333, 0.22266667, 0.052     ])
    """

    name = "constant_regret"

    def __init__(self, model, rewards=None, fading=None, max_edge_queue=None, **kwargs):

        if rewards is not None:
            rewards = np.array(rewards)
        else:
            rewards = model.incidence.sum(axis=0)
        self.rewards = model.normalize_rewards(rewards)

        if fading is None:
            fading = lin_fading
        self.fading = fading

        self.max_edge_queue = max_edge_queue

        super().__init__(model, **kwargs)

    def set_internal(self):
        super().set_internal()
        self.internal["scores"] = self.rewards
        self.internal["fading"] = self.fading
        self.internal["ready_edges"] = np.zeros(self.model.m, dtype=np.bool_)
        meq = self.max_edge_queue
        if meq is None:
            meq = 10 * self.max_queue
        self.internal["edge_queue"] = MultiQueue(self.model.m, max_queue=meq)

    def run(self):
        cr_core(logs=self.logs, **self.internal)
