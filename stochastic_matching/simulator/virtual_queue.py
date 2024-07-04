from numba import njit
import numpy as np

from stochastic_matching.simulator.simulator import Simulator
from stochastic_matching.simulator.multiqueue import MultiQueue


@njit
def vq_core(arrivals, graph, n_steps, queue_size,  # Generic arguments
            scores, ready_edges, edge_queue, forbidden_edges, threshold,  # Longest specific arguments
            traffic, queue_log, steps_done):  # Monitored variables
    """
    Jitted function for virtual-queue policy.

    Parameters
    ----------
    arrivals: :class:`~stochastic_matching.simulator.arrivals.Arrivals`
        Item arrival process.
    graph: :class:`~stochastic_matching.simulator.graph.Graph`
        Model graph.
    n_steps: :class:`int`
        Number of arrivals to process.
    queue_size: :class:`~numpy.ndarray`
        Number of waiting items of each class.
    scores: :class:`~numpy.ndarray`
        Default value for edges. Enables EGPD-like selection mechanism.
    ready_edges: :class:`~numpy.ndarray`
        Tells if given edge is actionable.
    edge_queue: :class:`~stochastic_matching.simulator.multiqueue.MultiQueue`
        Edges waiting to be matched.
    forbidden_edges: :class:`list`, optional
        Edges that are disabled.
    threshold: :class:`int`, optional
        Queue size above which forbidden edges become available again.
    traffic: :class:`~numpy.ndarray`
        Monitor traffic on edges.
    queue_log: :class:`~numpy.ndarray`
        Monitor queue sizes.
    steps_done: :class:`int`
        Number of arrivals processed so far.

    Returns
    -------
    :class:`int`
        Total number of steps performed, possibly including previous runs.
    """

    n, max_queue = queue_log.shape
    m = len(scores)

    # Optimize forbidden edges and set greedy flag.
    if forbidden_edges is not None:
        forbid = {k: True for k in forbidden_edges}
    else:
        forbid = {k: True for k in range(0)}

    infinity = edge_queue.infinity

    for age in range(n_steps):

        for j in range(n):
            queue_log[j, queue_size[j]] += 1

        # Draw an arrival
        node = arrivals.draw()
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return steps_done + age + 1

        # update scores
        scores[graph.edges(node)] += 1

        # update readyness
        if queue_size[node] == 1:
            for e in graph.edges(node):
                for v in graph.nodes(e):
                    if v != node and queue_size[v] == 0:
                        ready_edges[e] = False
                        break
                else:
                    ready_edges[e] = True

        if forbid is None:
            restrain = False
        else:
            restrain = True
            if threshold is not None:
                for v in range(n):
                    if queue_size[v] >= threshold:
                        restrain = False
                        break

        # Select best edge
        best_edge = -1
        best_score = 0

        for e in range(m):
            if restrain and e in forbid:
                continue
            score = scores[e]
            if score > best_score:
                best_edge = e
                best_score = score

        if best_edge > -1:
            # add edge to virtual queue
            edge_queue.add(best_edge, age)
            # Virtual pop of items:
            # for each node of the edge, lower scores of all adjacent edges by one
            for v in graph.nodes(best_edge):
                scores[graph.edges(v)] -= 1

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
            traffic[best_edge] += 1
            for v in graph.nodes(best_edge):
                queue_size[v] -= 1
                if queue_size[v] == 0:
                    ready_edges[graph.edges(v)] = False

    return steps_done + age + 1  # Return the updated number of steps achieved.


class VirtualQueue(Simulator):
    """
    Non-Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.simulator.Simulator`.
    Always pick-up the best edge according to a scoring function, even if that edge cannot be used (yet).

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    egpd_weights: :class:`list` or :class:`~numpy.ndarray`, optional
        Target rewards on edges.
    egpd_beta: :class:`float`
        Stabilization parameter. Close to 0, reward maximization is better but queues are more full.
    forbidden_edges: :class:`list` or :class:`~numpy.ndarray`, optional
        Egdes that should not be used.
    weights: :class:`list` or :class:`~numpy.ndarray`, optional
        Target rewards on edges. If weights are given, the forbidden edges are computed to match the target
        (overrides forbidden_edges argument).
    threshold: :class:`int`, optional
        Limit on queue size to apply edge interdiction (enforce stability on injective-only vertices).
    max_edge_queue: :class:`int`, optional
        In some extreme situation, the default allocated space for the edge virtual queue may be too small.
        If that happens someday, use this parameter to increase the VQ allocated memory.
    **kwargs
        Keyword arguments.


    Examples
    --------

    Let start with a working triangle. One can notice the results are different from the ones common to all
    greedy simulator.

    >>> import stochastic_matching as sm
    >>> sim = VirtualQueue(sm.Cycle(rates=[3, 4, 5]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'traffic': array([125, 162, 213]),
    'queue_log': array([[837, 105,  41,  13,   3,   1,   0,   0,   0,   0],
           [784, 131,  53,  22,   8,   2,   0,   0,   0,   0],
           [629, 187,  92,  51,  24,   9,   5,   3,   0,   0]]),
    'steps_done': 1000}

    Unstable diamond (simulation ends before completion due to drift).

    >>> sim = VirtualQueue(sm.CycleChain(rates='uniform'), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'traffic': array([34, 42,  7, 41, 36]),
    'queue_log': array([[142,  55,  34,  25,  23,  16,  24,  12,   7,   1],
           [319,  16,   3,   1,   0,   0,   0,   0,   0,   0],
           [317,  17,   4,   1,   0,   0,   0,   0,   0,   0],
           [107,  67,  64,  37,  22,  24,   7,   3,   6,   2]]),
    'steps_done': 339}

    Stable diamond without reward optimization:

    >>> sim = VirtualQueue(sm.CycleChain(rates=[1, 2, 2, 1]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'traffic': array([ 95,  84, 161,  84,  75]),
    'queue_log': array([[823, 120,  38,  19,   0,   0,   0,   0,   0,   0],
           [626, 215,  75,  46,  28,   9,   1,   0,   0,   0],
           [686, 212,  70,  24,   7,   1,   0,   0,   0,   0],
           [823, 118,  39,  12,   4,   4,   0,   0,   0,   0]]),
    'steps_done': 1000}

    Let's optimize (kill traffic on first and last edges).

    >>> sim = VirtualQueue(sm.CycleChain(rates=[1, 2, 2, 1]), egpd_weights=[0, 1, 1, 1, 0], n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'traffic': array([ 0, 22, 24, 27,  0]),
    'queue_log': array([[137,  16,   3,   0,   0,   0,   0,   0,   0,   0],
           [113,  14,   8,   6,   7,   4,   3,   1,   0,   0],
           [ 74,  21,  13,   7,   9,   8,   4,   5,  12,   3],
           [106,  31,  11,   5,   2,   1,   0,   0,   0,   0]]),
    'steps_done': 156}

    OK, it's working but we reached the maximal queue size quite fast. Let's reduce the pressure.

    >>> sim = VirtualQueue(sm.CycleChain(rates=[1, 2, 2, 1]), egpd_weights=[0, 1, 1, 1, 0], egpd_beta=.8, n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'traffic': array([ 32, 146, 161, 146,  13]),
    'queue_log': array([[692, 221,  68,  18,   1,   0,   0,   0,   0,   0],
           [515, 153, 123, 109,  48,  32,  16,   4,   0,   0],
           [662, 136,  91,  62,  38,   4,   2,   2,   2,   1],
           [791, 128,  50,  18,   7,   6,   0,   0,   0,   0]]), 'steps_done': 1000}

    A stable candy. While candies are not good for greedy policies, the virtual queue is
    designed to deal with it.

    >>> sim = VirtualQueue(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'traffic': array([109,  29,  17,  59,  58,  62, 107]),
    'queue_log': array([[302,  83,  93,  83,  54,  60,  43,  48,  41,  32,  49,  60,  31,
              3,   8,   7,   3,   0,   0,   0,   0,   0,   0,   0,   0],
           [825, 101,  27,  14,  26,   7,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [900,  63,  20,   8,   9,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [111,  49,  93, 108, 120, 101,  48,  28,  53,  32,   7,  30,  16,
             42,  28,  34,  31,  30,  26,  11,   2,   0,   0,   0,   0],
           [276, 185, 151,  95,  60,  40,  49,  52,  47,  33,  12,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [755, 142,  67,  36,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [710, 228,  47,  12,   3,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]),
    'steps_done': 1000}

    Last but not least: Stolyar's example from https://arxiv.org/abs/1608.01646

    >>> stol = sm.Model(incidence=[[1, 0, 0, 0, 1, 0, 0],
    ...                       [0, 1, 0, 0, 1, 1, 1],
    ...                       [0, 0, 1, 0, 0, 1, 1],
    ...                       [0, 0, 0, 1, 0, 0, 1]], rates=[1.2, 1.5, 2, .8])

    Without optimization, all we do is self-matches (no queue at all):

    >>> sim = VirtualQueue(stol, n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs['traffic']
    array([236, 279, 342, 143,   0,   0,   0])
    >>> sim.compute_average_queues()
    array([0., 0., 0., 0.])

    >>> rewards = [-1, -1, 1, 2, 5, 4, 7]

    With optimization, we get the desired results at the price of a huge queue:

    >>> sim = VirtualQueue(stol, egpd_weights=rewards, n_steps=3000, seed=42, max_queue=300)
    >>> sim.run()
    >>> sim.logs['traffic']
    array([  0,   0, 761, 242, 591,   0, 205])
    >>> sim.compute_average_queues()
    array([86.13933333,  0.53233333, 92.99666667,  0.33533333])

    Same traffic could be achieved with much lower queues by enforcing forbidden edges:

    >>> sim = VirtualQueue(stol, weights=rewards, n_steps=3000, seed=42, max_queue=300)
    >>> sim.run()
    >>> sim.logs['traffic']
    array([  0,   0, 961, 342, 691,   0, 105])
    >>> sim.compute_average_queues()
    array([2.731     , 0.69633333, 0.22266667, 0.052     ])
    """
    name = 'virtual_queue'

    def __init__(self, model, egpd_weights=None, egpd_beta=.01, forbidden_edges=None, weights=None, threshold=None,
                 max_edge_queue = None,
                 **kwargs):
        self.egpd_weights = np.array(egpd_weights) if egpd_weights is not None else None
        self.egpd_beta = egpd_beta
        if weights is not None:
            weights = np.array(weights)
            flow = model.optimize_rates(weights)
            forbidden_edges = [i for i in range(model.m) if flow[i] == 0]
            if len(forbidden_edges) == 0:
                forbidden_edges = None

        self.forbidden_edges = forbidden_edges
        self.threshold = threshold
        self.max_edge_queue = max_edge_queue if max_edge_queue is not None else 10*kwargs['max_queue']

        super().__init__(model, **kwargs)

    def set_internal(self):
        super().set_internal()
        if self.egpd_weights is None:
            self.internal['scores'] = np.zeros(self.model.m, dtype=int)
        else:
            self.internal['scores'] = self.egpd_weights / self.egpd_beta
        self.internal['edge_queue'] = MultiQueue(self.model.m, max_queue=self.max_edge_queue)
        self.internal['ready_edges'] = np.zeros(self.model.m, dtype=np.bool_)
        self.internal['forbidden_edges'] = self.forbidden_edges
        self.internal['threshold'] = self.threshold

    def run(self):
        self.logs['steps_done'] = vq_core(**self.internal, **self.logs)
