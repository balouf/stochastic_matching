from numba import njit
import numpy as np
from stochastic_matching.simulator.simulator import Simulator


@njit
def random_item_selector(graph, queue_size, node):
    """
    Selects a feasible edge at random proportionally to the number of waiting items.
    """
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


class RandomItem(Simulator):
    """
    Greedy matching simulator derived from :class:`~stochastic_matching.simulator.simulator.Simulator`.
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
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [276 342 382]
    Traffic: [118 158 224]
    Queues: [[865  92  32  10   1   0   0   0   0   0   0]
     [750 142  62  28  12   3   2   1   0   0   0]
     [662 164  73  36  21   7  10  12   8   5   2]]
    Steps done: 1000

    An ill braess graph (simulation ends before completion due to drift).

    >>> sim = RandomItem(sm.CycleChain(rates='uniform'), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [25 17 13 12]
    Traffic: [10  6  2  5  5]
    Queues: [[ 9  4  7  7  5  4  9  8 11  3]
     [67  0  0  0  0  0  0  0  0  0]
     [60  4  3  0  0  0  0  0  0  0]
     [15 18 16  9  4  5  0  0  0  0]]
    Steps done: 67

    A working candy (but candies are not good for greedy policies).

    >>> sim = RandomItem(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [112 106 166 105 206 112 115]
    Traffic: [66 46 39 61 64 51 81]
    Queues: [[683 125  63  25   8  13   2   3   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [659 130  65  34  32   2   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [757  89  36  19  12   9   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [ 23  10  23  30  28  56  56 109  81  85 114  55  22  38  17  36  32  30
        7   4   3   5  13  20  25]
     [673 123  73  28  19   6   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [769 112  33   4   4   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [708 122  55  17   4   5   5   5   1   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]]
    Steps done: 922
    """
    name = "random_item"

    def set_internal(self):
        super().set_internal()
        self.internal['selector'] = random_item_selector
