from numba import njit
import numpy as np
from stochastic_matching.newsimulator.queue_size import QueueSize, make_qs_core


@njit
def random_edge_selector(graph, queue_size, node):
    best_edge = -1
    best_score = -1.0
    for e in graph.edges(node):
        for v in graph.nodes(e):
            if queue_size[v] == 0:
                break
        else:
            score = np.random.rand()
            if score > best_score:
                best_score = score
                best_edge = e
    return best_edge


class RandomEdge(QueueSize):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.queue_size.QueueSize`.
    When multiple choices are possible, one edge is chosen uniformly at random.

    Parameters
    ----------

    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    **kwargs
        Keyword arguments.


    Examples
    --------

    Let start with a working triangle. One can notice the results are not the than for other greedy simulators.
    This may seem off because there are no multiple choices in a triangle
    (always one non-empty queue at most under a greedy policy).
    The root cause is that the arrival process shares the same random generator than the random selection.

    >>> import stochastic_matching as sm
    >>> triangle = sm.Cycle(rates=[3, 4, 5])
    >>> sim = RandomEdge(triangle, n_steps=1000, seed=42, max_queue=11)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([118, 158, 224]),
    'queue_log': array([[865,  92,  32,  10,   1,   0,   0,   0,   0,   0,   0],
           [750, 142,  62,  28,  12,   3,   2,   1,   0,   0,   0],
           [662, 164,  73,  36,  21,   7,  10,  12,   8,   5,   2]]),
    'steps_done': 1000}

     Sanity check: results are unchanged if the graph is treated as hypergraph.

    >>> triangle.adjacency = None
    >>> sim = RandomEdge(triangle, n_steps=1000, seed=42, max_queue=11)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([118, 158, 224]),
    'queue_log': array([[865,  92,  32,  10,   1,   0,   0,   0,   0,   0,   0],
           [750, 142,  62,  28,  12,   3,   2,   1,   0,   0,   0],
           [662, 164,  73,  36,  21,   7,  10,  12,   8,   5,   2]]),
    'steps_done': 1000}

    A ill diamond graph (simulation ends before completion due to drift).

    >>> sim = RandomEdge(sm.CycleChain(rates='uniform'), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([10,  8,  2,  8,  6]),
    'queue_log': array([[10,  4,  7, 13,  5,  5, 10, 24,  5,  1],
           [84,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [77,  4,  3,  0,  0,  0,  0,  0,  0,  0],
           [14,  8, 15, 19, 14,  8,  2,  1,  2,  1]]),
    'steps_done': 84}
    >>> sim.compute_flow()
    array([0.47619048, 0.38095238, 0.0952381 , 0.38095238, 0.28571429])

    A working candy (but candies are not good for greedy policies).

    >>> sim = RandomEdge(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([24, 22, 19, 28, 35, 19, 26]),
    'queue_log': array([[306,  77,  15,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [306,  53,  32,   9,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [342,  36,  18,   5,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  8,   6,   9,  13,   7,  34,  41,  47,  23,   9,   2,   3,   2,
              7,   6,   2,  12,  46,  16,  14,  39,  43,   2,   5,   5],
           [276,  51,  38,  30,   6,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [325,  50,  21,   4,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [306,  47,  23,   9,   2,   3,   5,   5,   1,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]),
    'steps_done': 401}

    Note that you can reset the simulator before starting another run.

    >>> sim.reset()
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([24, 22, 19, 28, 35, 19, 26]),
    'queue_log': array([[306,  77,  15,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [306,  53,  32,   9,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [342,  36,  18,   5,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  8,   6,   9,  13,   7,  34,  41,  47,  23,   9,   2,   3,   2,
              7,   6,   2,  12,  46,  16,  14,  39,  43,   2,   5,   5],
           [276,  51,  38,  30,   6,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [325,  50,  21,   4,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [306,  47,  23,   9,   2,   3,   5,   5,   1,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]),
    'steps_done': 401}

    You can display the distribution of queue sizes as a ccdf:

    >>> sim.show_ccdf() # doctest: +ELLIPSIS
    <Figure size ...x... with 1 Axes>
    """
    name = "random_edge"

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.core = make_qs_core(random_edge_selector)
