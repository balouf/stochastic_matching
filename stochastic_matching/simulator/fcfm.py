from numba import njit

from stochastic_matching.simulator.simulator import Simulator
from stochastic_matching.simulator.multiqueue import MultiQueue


@njit
def fcfm_core(logs, arrivals, graph, n_steps, queue_size, queues):
    """
    Jitted function for first-come, first-matched policy.

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
    queues: :class:`~stochastic_matching.simulator.multiqueue.MultiQueue`
        Waiting items of each class.

    Returns
    -------
    None
    """

    n, max_queue = logs.queue_log.shape
    inf = queues.infinity

    for age in range(n_steps):

        node = arrivals.draw()
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return None
        queues.add(node, age)

        # Test if an actionable edge may be present
        best_edge = -1
        if queue_size[node] == 1:
            best_age = inf
            # check oldest edge node
            for e in graph.edges(node):
                edge_age = inf
                for v in graph.nodes(e):
                    v_age = queues.oldest(v)
                    if v_age == inf:
                        break
                    edge_age = min(edge_age, v_age)
                else:
                    if edge_age < best_age:
                        best_edge = e
                        best_age = edge_age

            if best_edge > -1:
                # logs.traffic[best_edge] += 1
                queue_size[graph.nodes(best_edge)] -= 1
                for v in graph.nodes(best_edge):
                    queues.pop(v)

        logs.update(queue_size=queue_size, node=node, edge=best_edge)


class FCFM(Simulator):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.simulator.Simulator`.
    When multiple choices are possible, the oldest item is chosen.

    Examples
    --------

    Let start with a working triangle. One can notice the results are the same for all greedy simulator because
    there are no multiple choices in a triangle (always one non-empty queue at most under a greedy policy).

    >>> import stochastic_matching as sm
    >>> sim = FCFM(sm.Cycle(rates=[3, 4, 5]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [287 338 375]
    Traffic: [125 162 213]
    Queues: [[838 104  41  13   3   1   0   0   0   0]
     [796 119  53  22   8   2   0   0   0   0]
     [640 176  92  51  24   9   5   3   0   0]]
    Steps done: 1000

    Unstable diamond (simulation ends before completion due to drift).

    >>> sim = FCFM(sm.CycleChain(rates=[1, 1, 1, 1]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [85 82 85 86]
    Traffic: [34 42  7 41 36]
    Queues: [[126  70  22  26  29  12  23  15  10   5]
     [326   8   3   1   0   0   0   0   0   0]
     [321  12   4   1   0   0   0   0   0   0]
     [105  80  65  28  31  15   4   2   6   2]]
    Steps done: 338

    A stable candy (but candies are not good for greedy policies).

    >>> sim = FCFM(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]),
    ...            n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [46 26 32 37 70 35 45]
    Traffic: [24 17  2 23 33 12 13]
    Queues: [[ 23  32  45  38  22  43  31  34  20   3   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [290   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [290   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [ 9   1   7   9   3   3  26  37   4   8  10   9   2  10  40  11   2  16
        3   3  21  27  22   1   7]
     [212  49  22   5   3   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [233  41   6   7   4   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [231  33  16   4   6   1   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]]
    Steps done: 291
    """
    name = 'fcfm'

    def set_internal(self):
        super().set_internal()
        self.internal['queues'] = MultiQueue(self.model.n, max_queue=self.max_queue + 1)

    def run(self):
        fcfm_core(logs=self.logs, **self.internal)
