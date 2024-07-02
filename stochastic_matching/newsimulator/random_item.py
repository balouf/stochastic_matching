from numba import njit
import numpy as np
from stochastic_matching.newsimulator.queue_size import QueueSize, make_qs_core


@njit
def random_item_selector(graph, queue_size, node):
    best_edge = -1
    prev_weight = 0.0

    for e in graph.edges(node):
        weight = 0.0
        for v in graph.nodes(e):
            w = queue_size[v]
            if w == 0:
                break
            weight += w
        else:
            prev_weight += weight
            if prev_weight * np.random.rand() < weight:
                best_edge = e
    return best_edge


class RandomItem(QueueSize):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.classes.QueueSizeSimulator`.
    When multiple choices are possible, chooses proportionally to the sizes of the queues
    (or sum of queues for hyperedges).

    Parameters
    ----------

    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    **kwargs
        Keyword arguments.

    Examples
    --------

    Let start with a working triangle.

    >>> import stochastic_matching as sm
    >>> sim = RandomItem(sm.Cycle(rates=[3, 4, 5]), n_steps=1000, seed=42, max_queue=11)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([118, 158, 224]),
    'queue_log': array([[865,  92,  32,  10,   1,   0,   0,   0,   0,   0,   0],
           [750, 142,  62,  28,  12,   3,   2,   1,   0,   0,   0],
           [662, 164,  73,  36,  21,   7,  10,  12,   8,   5,   2]]),
    'steps_done': 1000}

    A ill braess graph (simulation ends before completion due to drift).

    >>> sim = RandomItem(sm.CycleChain(rates='uniform'), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([10,  6,  2,  5,  5]),
     'queue_log': array([[10,  4,  7,  7,  5,  4,  9,  8, 11,  3],
           [68,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [61,  4,  3,  0,  0,  0,  0,  0,  0,  0],
           [16, 18, 16,  9,  4,  5,  0,  0,  0,  0]]),
     'steps_done': 68}

    A working candy (but candies are not good for greedy policies).

    >>> sim = RandomItem(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([66, 46, 39, 61, 64, 51, 81]),
    'queue_log': array([[684, 125,  63,  25,   8,  13,   2,   3,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [660, 130,  65,  34,  32,   2,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [758,  89,  36,  19,  12,   9,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 24,  10,  23,  30,  28,  56,  56, 109,  81,  85, 114,  55,  22,
             38,  17,  36,  32,  30,   7,   4,   3,   5,  13,  20,  25],
           [674, 123,  73,  28,  19,   6,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [770, 112,  33,   4,   4,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [709, 122,  55,  17,   4,   5,   5,   5,   1,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]),
    'steps_done': 923}
    """
    name = "random_item"

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.core = make_qs_core(random_item_selector)
