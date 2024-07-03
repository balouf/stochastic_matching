import numpy as np
from numba import njit

from stochastic_matching.simulator.simulator import Simulator


def make_priority_selector(weights, threshold, counterweights):
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
