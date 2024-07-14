from numba import njit
import numpy as np

from stochastic_matching.simulator.extended import ExtendedSimulator
from stochastic_matching.simulator.multiqueue import MultiQueue


@njit
def vq_core(logs, arrivals, graph, n_steps, queue_size,  # Generic arguments
            scores, ready_edges, edge_queue, forbidden_edges, k):  # Longest specific arguments
    """
    Jitted function for virtual-queue policy.

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
        Default value for edges. Enables EGPD-like selection mechanism.
    ready_edges: :class:`~numpy.ndarray`
        Tells if given edge is actionable.
    edge_queue: :class:`~stochastic_matching.simulator.multiqueue.MultiQueue`
        Edges waiting to be matched.
    forbidden_edges: :class:`list`, optional
        Edges that are disabled.
    k: :class:`int`, optional
        Queue size above which forbidden edges become available again.

    Returns
    -------
    :class:`int`
        Total number of steps performed, possibly including previous runs.
    """

    n, max_queue = logs.queue_log.shape
    m = len(scores)

    # Optimize forbidden edges and set greedy flag.
    if forbidden_edges is not None:
        forbid = {k: True for k in forbidden_edges}
        has_forbidden = True
    else:
        forbid = {k: True for k in range(0)}
        has_forbidden = False

    infinity = edge_queue.infinity

    for age in range(n_steps):

        # Draw an arrival
        node = arrivals.draw()
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return None

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

        if has_forbidden:
            restrain = True
            if k is not None:
                for v in range(n):
                    if queue_size[v] >= k:
                        restrain = False
                        break
        else:
            restrain = False

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
            for v in graph.nodes(best_edge):
                queue_size[v] -= 1
                if queue_size[v] == 0:
                    ready_edges[graph.edges(v)] = False

        logs.update(queue_size=queue_size, node=node, edge=best_edge)


class VirtualQueue(ExtendedSimulator):
    """
    Non-Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.extended.ExtendedSimulator`.
    Always pick-up the best edge according to a scoring function, even if that edge cannot be used (yet).

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    max_edge_queue: :class:`int`, optional
        In some extreme situation, the default allocated space for the edge virtual queue may be too small.
        If that happens someday, use this parameter to increase the VQ allocated memory.
    **kwargs
        Keyword parameters of :class:`~stochastic_matching.simulator.extended.ExtendedSimulator`.


    Examples
    --------

    Let start with a working triangle.

    >>> import stochastic_matching as sm
    >>> sim = VirtualQueue(sm.Cycle(rates=[3, 4, 5]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [287 338 375]
    Traffic: [125 162 213]
    Queues: [[837 105  41  13   3   1   0   0   0   0]
     [784 131  53  22   8   2   0   0   0   0]
     [629 187  92  51  24   9   5   3   0   0]]
    Steps done: 1000

    Unstable diamond (simulation ends before completion due to drift).

    >>> sim = VirtualQueue(sm.CycleChain(rates='uniform'), n_steps=1000, seed=42, max_queue=10)
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

    >>> sim = VirtualQueue(sm.CycleChain(rates=[1, 2, 2, 1]), n_steps=1000, seed=42, max_queue=10)
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

    >>> sim = VirtualQueue(sm.CycleChain(rates=[1, 2, 2, 1]), rewards=[0, 1, 1, 1, 0], beta=.01,
    ...                    n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [22 51 55 27]
    Traffic: [ 0 22 24 27  0]
    Queues: [[136  16   3   0   0   0   0   0   0   0]
     [112  14   8   6   7   4   3   1   0   0]
     [ 73  21  13   7   9   8   4   5  12   3]
     [105  31  11   5   2   1   0   0   0   0]]
    Steps done: 155

    OK, it's working, but we reached the maximal queue size quite fast. Let's reduce the pressure.

    >>> sim = VirtualQueue(sm.CycleChain(rates=[1, 2, 2, 1]), rewards=[0, 1, 1, 1, 0],
    ...                    beta=.8, n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [179 342 320 159]
    Traffic: [ 32 146 161 146  13]
    Queues: [[691 222  68  18   1   0   0   0   0   0]
     [514 153 123 110  48  32  16   4   0   0]
     [662 136  91  62  38   4   2   2   2   1]
     [791 128  50  18   7   6   0   0   0   0]]
    Steps done: 1000

    Let's now use a k-filtering for this diamond:

    >>> sim = VirtualQueue(sm.CycleChain(rates=[1, 2, 2, 1]), forbidden_edges=True,
    ...                    rewards=[0, 1, 1, 1, 0], n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [27 57 65 31]
    Traffic: [ 0 27 29 28  0]
    Queues: [[158  16   3   3   0   0   0   0   0   0]
     [136  19   8   4   5   4   3   1   0   0]
     [ 70  25  13  14  12  15  21   8   1   1]
     [ 96  24  13  12   9  16   8   2   0   0]]
    Steps done: 180

    Let us reduce the pressure.

    >>> sim = VirtualQueue(sm.CycleChain(rates=[1, 2, 2, 1]), forbidden_edges=True, rewards=[0, 1, 1, 1, 0],
    ...                    n_steps=1000, seed=42, max_queue=10, k=6)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [179 342 320 159]
    Traffic: [ 31 145 161 145  14]
    Queues: [[600 171 122  81  20   6   0   0   0   0]
     [444 151 137 104  81  65  14   4   0   0]
     [698  83  52  59  57  38   7   3   2   1]
     [730 130  64  35  26   9   2   4   0   0]]
    Steps done: 1000

    A stable candy. While candies are not good for greedy policies, the virtual queue is
    designed to deal with it.

    >>> sim = VirtualQueue(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), n_steps=1000, seed=42, max_queue=25)
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

    >>> stol = sm.Model(incidence=[[1, 0, 0, 0, 1, 0, 0],
    ...                       [0, 1, 0, 0, 1, 1, 1],
    ...                       [0, 0, 1, 0, 0, 1, 1],
    ...                       [0, 0, 0, 1, 0, 0, 1]], rates=[1.2, 1.5, 2, .8])

    Without optimization, all we do is self-matches (no queue at all):

    >>> sim = VirtualQueue(stol, n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs.traffic.astype(int)
    array([236, 279, 342, 143,   0,   0,   0])
    >>> sim.avg_queues
    array([0., 0., 0., 0.])

    >>> rewards = [-1, -1, 1, 2, 5, 4, 7]

    With optimization, we get the desired results at the price of a huge queue:

    >>> sim = VirtualQueue(stol, rewards=rewards, beta=.01, n_steps=3000, seed=42, max_queue=300)
    >>> sim.run()
    >>> sim.logs.traffic.astype(int)
    array([  0,   0, 761, 242, 591,   0, 205])
    >>> sim.avg_queues
    array([86.17266667,  0.53233333, 93.03      ,  0.33533333])

    Same traffic could be achieved with much lower queues by enforcing forbidden edges:

    >>> sim = VirtualQueue(stol, rewards=rewards, forbidden_edges=True, n_steps=3000, seed=42, max_queue=300)
    >>> sim.run()
    >>> sim.logs.traffic.astype(int)
    array([  0,   0, 961, 342, 691,   0, 105])
    >>> sim.avg_queues
    array([2.731     , 0.69633333, 0.22266667, 0.052     ])
    """
    name = 'virtual_queue'

    def __init__(self, model, max_edge_queue=None, **kwargs):
        self.max_edge_queue = max_edge_queue
        super().__init__(model, **kwargs)

    def set_internal(self):
        super().set_internal()
        self.internal['ready_edges'] = np.zeros(self.model.m, dtype=np.bool_)
        meq = self.max_edge_queue
        if meq is None:
            meq = 10 * self.max_queue
        self.internal['edge_queue'] = MultiQueue(self.model.m, max_queue=meq)

    def run(self):
        vq_core(logs=self.logs, **self.internal)
