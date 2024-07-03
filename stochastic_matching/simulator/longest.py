from numba import njit
import numpy as np

from stochastic_matching.simulator.simulator import Simulator


@njit
def longest_core(arrivals, graph, n_steps, queue_size,  # Generic arguments
                 scores, forbidden_edges, threshold,  # Longest specific arguments
                 trafic, queue_log, steps_done):  # Monitored variables

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

        # Test if an activable edge may be present
        if not greedy or queue_size[node] == 1:
            # Should we activate edge restriction?
            if forbid is None:
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
                trafic[best_edge] += 1
                queue_size[graph.nodes(best_edge)] -= 1

    return steps_done + age + 1


from stochastic_matching.simulator.simulator import Simulator


class Longest(Simulator):
    """
    Examples
    --------

        Let start with a working triangle. Not that the results are the same for all greedy old_simulator because
    there are no decision in a triangle (always at most one non-empty queue under a greedy policy).

    >>> import stochastic_matching as sm
    >>> sim = Longest(sm.Cycle(rates=[3, 4, 5]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213]),
    'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
       [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
       [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]]),
    'steps_done': 1000}

    A non stabilizable diamond (simulation ends before completion due to drift).

    >>> sim = Longest(sm.CycleChain(rates='uniform'), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([38, 38,  7, 37, 40]),
    'queue_log': array([[127,  74,  28,  37,  21,  32,  16,   1,   2,   1],
           [327,   8,   3,   1,   0,   0,   0,   0,   0,   0],
           [322,  12,   4,   1,   0,   0,   0,   0,   0,   0],
           [ 91,  80,  47,  37,  37,  23,  11,   3,   5,   5]]),
    'steps_done': 339}

    A stabilizable candy (but candies are not good for greedy policies).

    >>> sim = Longest(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([24, 17,  2, 23, 33, 12, 13]),
    'queue_log': array([[ 24,  32,  45,  38,  22,  43,  31,  34,  20,   3,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [291,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [291,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 10,   1,   7,   9,   3,   3,  26,  37,   4,   8,  10,   9,   2,
             10,  40,  11,   2,  16,   3,   3,  21,  27,  22,   1,   7],
           [213,  49,  22,   5,   3,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [234,  41,   6,   7,   4,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [232,  33,  16,   4,   6,   1,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]),
    'steps_done': 292}
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
        self.logs['steps_done'] = longest_core(**self.internal, **self.logs)
