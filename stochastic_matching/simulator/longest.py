from numba import njit
import numpy as np

from stochastic_matching.simulator.simulator import Simulator


@njit
def longest_core(arrivals, graph, n_steps, queue_size,  # Generic arguments
                 scores, forbidden_edges, threshold,  # Longest specific arguments
                 traffic, queue_log, steps_done):  # Monitored variables
    """
    Generic jitted function for queue-size based policies.

    Parameters
    ----------
    arrivals: :class:`~stochastic_matching.simulator.arrivals.Arrivals`
        Item arrival process.
    graph: :class:`~stochastic_matching.simulator.graph.JitHyperGraph`
        Model graph.
    n_steps: :class:`int`
        Number of arrivals to process.
    queue_size: :class:`~numpy.ndarray`
        Number of waiting items of each class.
    scores: :class:`~numpy.ndarray`
        Default value for edges. Enables EGPD-like selection mechanism.
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

    # Optimize forbidden edges and set greedy flag.
    if forbidden_edges is not None:
        forbid = {k: True for k in forbidden_edges}
        greedy = False
    else:
        forbid = {k: True for k in range(0)}
        greedy = True

    for age in range(n_steps):

        for j in range(n):
            queue_log[j, queue_size[j]] += 1

        # Draw an arrival
        node = arrivals.draw()
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return steps_done + age + 1

        # Test if an actionable edge may be present
        if not greedy or queue_size[node] == 1:
            # Should we activate edge restriction?
            if greedy:
                restrain = False
            else:
                restrain = True
                if threshold is not None:
                    for v in range(n):
                        if queue_size[v] >= threshold:
                            restrain = False
                            break

            best_edge = -1
            best_score = -1
            # update scores
            for e in graph.edges(node):
                if restrain and e in forbid:
                    continue
                score = scores[e]
                for v in graph.nodes(e):
                    w = queue_size[v]
                    if w == 0:
                        break
                    if v != node:
                        score += w
                else:
                    if score > best_score:
                        best_edge = e
                        best_score = score

            if best_edge > -1:
                traffic[best_edge] += 1
                queue_size[graph.nodes(best_edge)] -= 1

    return steps_done + age + 1


class Longest(Simulator):
    """
    Greedy Matching old_simulator derived from :class:`~stochastic_matching.simulator.simulator.Simulator`.
    When multiple choices are possible, the longest queue (or sum of queues for hyperedges) is chosen.

    Parameters
    ----------

    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    egpd_weights: :class:`~numpy.ndarray` or :class:`list`, optional
        EGPD rewards for default edge score.
    egpd_beta: :class:`float`, default=.01
        EGPD factor (smaller means higher reward impact).
    forbidden_edges: :class:`list` or :class:`~numpy.ndarray`, optional
        Edges that should not be used.
    weights: :class:`~numpy.ndarray`, optional
        Target rewards on edges. If weights are given, the forbidden edges are computed to match the target
        (overrides forbidden_edges argument).
    threshold: :class:`int`, optional
        Limit on queue size to apply edge interdiction (enforce stability on injective-only vertices).
    **kwargs
        Keyword arguments.

    Examples
    --------

    Let's start with a working triangle. Not that the results are the same for all greedy old_simulator because
    there are no decision in a triangle (always at most one non-empty queue under a greedy policy).

    >>> import stochastic_matching as sm
    >>> sim = Longest(sm.Cycle(rates=[3, 4, 5]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    Traffic: [125 162 213]
    Queues: [[838 104  41  13   3   1   0   0   0   0]
     [796 119  53  22   8   2   0   0   0   0]
     [640 176  92  51  24   9   5   3   0   0]]
    Steps done: 1000

    A non stabilizable diamond (simulation ends before completion due to drift).

    >>> sim = Longest(sm.CycleChain(rates='uniform'), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    Traffic: [38 38  7 37 40]
    Queues: [[127  74  28  37  21  32  16   1   2   1]
     [327   8   3   1   0   0   0   0   0   0]
     [322  12   4   1   0   0   0   0   0   0]
     [ 91  80  47  37  37  23  11   3   5   5]]
    Steps done: 339

    A stabilizable candy (but candies are not good for greedy policies).

    >>> sim = Longest(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    Traffic: [24 17  2 23 33 12 13]
    Queues: [[ 24  32  45  38  22  43  31  34  20   3   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [291   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [291   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [ 10   1   7   9   3   3  26  37   4   8  10   9   2  10  40  11   2  16
        3   3  21  27  22   1   7]
     [213  49  22   5   3   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [234  41   6   7   4   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [232  33  16   4   6   1   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]]
    Steps done: 292

    Using Stolyar EGCD technique (longest version), one can try to reach some vertices.

    That works for the Fish graph:

    >>> fish = sm.KayakPaddle(m=4, l=0, rates=[4, 4, 3, 2, 3, 2])
    >>> sim = Longest(fish, egpd_weights=[0, 2, 2, 0, 1, 1, 0], n_steps=10000, seed=42)
    >>> sim.run()

    The flow avoids one edge:

    >>> sim.compute_flow()
    array([2.8764, 1.089 , 1.0008, 0.    , 0.8532, 2.079 , 1.0062])

    The price is one node having a big queue:

    >>> avg_queues = sim.compute_average_queues()
    >>> avg_queues[-1]
    61.1665

    Other nodes are not affected:

    >>> np.round(np.mean(avg_queues[:-1]), decimals=4)
    0.7361

    Playing with the beta parameter allows to adjust the trade-off (smaller queue, leak on the forbidden edge):

    >>> sim = Longest(fish, egpd_weights=[0, 2, 2, 0, 1, 1, 0], egpd_beta=.1, n_steps=10000, seed=42)
    >>> sim.run()
    >>> sim.compute_flow()
    array([2.9574, 1.008 , 0.9198, 0.018 , 0.9972, 2.061 , 1.0242])
    >>> sim.compute_average_queues()[-1]
    8.4616

    Alternatively, one can use the k-filtering techniques:

    >>> diamond = sm.CycleChain(rates=[1, 2, 2, 1])
    >>> diamond.run('longest', forbidden_edges=[0, 4], seed=42,
    ...                            threshold=100, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([0.   , 0.954, 0.966, 0.954, 0.   ])

    Same result can be achieved by putting low weights on 0 and 4.

    >>> diamond.run('longest', weights=[1, 2, 2, 2, 1], seed=42,
    ...                            threshold=100, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([0.   , 0.954, 0.966, 0.954, 0.   ])

    Note that if the threshold is too low some exceptions are done to limit the queue sizes.

    >>> diamond.run('longest', weights=[1, 2, 2, 2, 1], seed=42,
    ...                            threshold=10, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([0.108, 0.93 , 0.966, 0.93 , 0.024])

    Having no relaxation usually leads to large, possibly overflowing, queues.
    However, this doesn't show on small simulations.

    >>> diamond.run('longest', weights=[1, 2, 2, 2, 1], seed=42,
    ...                            n_steps=1000, max_queue=100)
    True
    >>> diamond.simulator.compute_average_queues()
    array([7.495, 7.797, 1.067, 1.363])
    >>> diamond.simulation
    array([0.   , 0.954, 0.966, 0.954, 0.   ])


    To compare with the priority-based pure greedy version:

    >>> diamond.run('priority', weights=[1, 2, 2, 2, 1], n_steps=1000, max_queue=1000, seed=42)
    True
    >>> diamond.simulation
    array([0.444, 0.63 , 0.966, 0.63 , 0.324])

    Another example with other rates.

    >>> diamond.rates=[4, 5, 2, 1]

    Optimize with the first and last edges that provide less reward.

    >>> diamond.run('longest', weights=[1, 2, 2, 2, 1], seed=42,
    ...                            threshold=100, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([3.264, 0.888, 0.948, 0.84 , 0.   ])

    Increase the reward on the first edge.

    >>> diamond.run('longest', weights=[4, 2, 2, 2, 1], seed=42,
    ...                            threshold=100, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([4.152, 0.   , 0.996, 0.   , 0.84 ])

    On bijective graphs, no edge is forbidden whatever the weights.

    >>> paw = sm.Tadpole()
    >>> paw.run('longest', weights=[6, 3, 1, 2], seed=42,
    ...                            threshold=100, n_steps=1000, max_queue=1000)
    True
    >>> paw.simulation
    array([1.048, 1.056, 1.016, 0.88 ])
    """
    name = 'longest'

    def __init__(self, model, egpd_weights=None, egpd_beta=.01, forbidden_edges=None, weights=None, threshold=None,
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
        super().__init__(model, **kwargs)

    def set_internal(self):
        super().set_internal()
        self.internal['scores'] = np.zeros(self.model.m,
                                           dtype=int) if self.egpd_weights is None else self.egpd_weights / self.egpd_beta
        self.internal['forbidden_edges'] = self.forbidden_edges
        self.internal['threshold'] = self.threshold

    def run(self):
        self.logs.steps_done = longest_core(**self.internal, **self.logs.asdict())
