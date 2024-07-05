from numba import njit

from stochastic_matching.simulator.simulator import Simulator
from stochastic_matching.simulator.multiqueue import MultiQueue


@njit
def fcfm_core(arrivals, graph, n_steps, queue_size,  # Generic arguments
              queues,  # FCFM specific argument
              traffic, queue_log, steps_done):  # Monitored variables
    """
    Jitted function for first-come, first-matched policy.

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
    queues: :class:`~stochastic_matching.simulator.multiqueue.MultiQueue`
        Waiting items of each class.
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
    inf = queues.infinity

    for age in range(n_steps):

        for j in range(n):
            queue_log[j, queue_size[j]] += 1

        node = arrivals.draw()
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return steps_done + age + 1
        queues.add(node, age)

        # Test if an actionable edge may be present
        if queue_size[node] == 1:
            best_edge = -1
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
                traffic[best_edge] += 1
                queue_size[graph.nodes(best_edge)] -= 1
                for v in graph.nodes(best_edge):
                    queues.pop(v)

    return steps_done + age + 1


class FCFM(Simulator):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.simulator.Simulator`.
    When multiple choices are possible, the oldest item is chosen.

    Examples
    --------

    Let start with a working triangle. One can notice the results are the same for all greedy old_simulator because
    there are no multiple choices in a triangle (always one non-empty queue at most under a greedy policy).

    >>> import stochastic_matching as sm
    >>> sim = FCFM(sm.Cycle(rates=[3, 4, 5]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    Traffic: [125 162 213]
    Queues: [[838 104  41  13   3   1   0   0   0   0]
     [796 119  53  22   8   2   0   0   0   0]
     [640 176  92  51  24   9   5   3   0   0]]
    Steps done: 1000

    Unstable diamond (simulation ends before completion due to drift).

    >>> sim = FCFM(sm.CycleChain(rates=[1, 1, 1, 1]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    Traffic: [34 42  7 41 36]
    Queues: [[127  70  22  26  29  12  23  15  10   5]
     [327   8   3   1   0   0   0   0   0   0]
     [322  12   4   1   0   0   0   0   0   0]
     [106  80  65  28  31  15   4   2   6   2]]
    Steps done: 339

    A stable candy (but candies are not good for greedy policies).

    >>> sim = FCFM(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]),
    ...            n_steps=1000, seed=42, max_queue=25)
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
    """
    name = 'fcfm'

    def set_internal(self):
        super().set_internal()
        self.internal['queues'] = MultiQueue(self.model.n, max_queue=self.max_queue + 1)

    def run(self):
        self.logs.steps_done = fcfm_core(**self.internal, **self.logs.asdict())
