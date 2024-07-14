import numpy as np
from numba import njit

from stochastic_matching.simulator.simulator import Simulator


def make_priority_selector(weights, threshold, counterweights):
    """
    Make a jitted edge selector based on priorities.
    """
    def priority_selector(graph, queue_size, node):
        best_edge = -1
        best_weight = -1000

        if threshold is None:
            w = weights
        else:
            for s in queue_size:
                if s >= threshold:
                    w = counterweights
                    break
            else:
                w = weights

        for e in graph.edges(node):
            for v in graph.nodes(e):
                if queue_size[v] == 0:
                    break
            else:
                if w[e] > best_weight:
                    best_weight = w[e]
                    best_edge = e
        return best_edge

    return njit(priority_selector)


class Priority(Simulator):
    """
    Greedy policy based on pre-determined preferences on edges.

    A threshold can be specified to alter the weights if the queue sizes get too big.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    weights: :class:`list` or :class:`~numpy.ndarray`
        Priorities associated to the edges.
    threshold: :class:`int`, optional
        Limit on max queue size to apply the weight priority.
    counterweights: :class:`list` or :class:`~numpy.ndarray`, optional
        Priority to use above threshold (if not provided, reverse weights is used).
    **kwargs
        Keyword arguments.

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> fish = sm.KayakPaddle(m=4, l=0, rates=[4, 4, 3, 2, 3, 2])
    >>> fish.run('priority', weights=[0, 2, 2, 0, 1, 1, 0],
    ...                          threshold=50, counterweights = [0, 0, 0, 1, 2, 2, 1],
    ...                          n_steps=10000, seed=42)
    True

    These priorities are efficient at stabilizing the policy while avoiding edge 3.

    >>> fish.simulation
    array([2.925 , 1.0404, 0.9522, 0.    , 0.9504, 2.0808, 1.0044])

    The last node is the pseudo-instable node.

    >>> fish.simulator.avg_queues[-1]
    38.346
    >>> np.round(np.mean(fish.simulator.avg_queues[:-1]), decimals=2)
    0.75

    Choosing proper counter-weights is important.

    >>> fish.run('priority', weights=[0, 2, 2, 0, 1, 1, 0],
    ...                          threshold=50,
    ...                          n_steps=10000, seed=42)
    True
    >>> fish.simulation
    array([2.9232, 1.0422, 0.9504, 0.216 , 0.7344, 1.8666, 1.2186])
    >>> fish.simulator.avg_queues[-1]
    38.6016
    """
    name = "priority"

    def __init__(self, model, weights, threshold=None, counterweights=None, **kwargs):
        weights = np.array(weights)
        if threshold is not None:
            if counterweights is None:
                counterweights = -weights
            else:
                counterweights = np.array(counterweights)

        self.weights = weights
        self.threshold = threshold
        self.counterweights = counterweights

        super().__init__(model, **kwargs)

    def set_internal(self):
        super().set_internal()
        self.internal['selector'] = make_priority_selector(weights=self.weights,
                                                           threshold=self.threshold,
                                                           counterweights=self.counterweights)
