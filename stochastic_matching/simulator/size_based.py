from abc import ABC

import numpy as np
from numba import njit

from stochastic_matching.simulator.generic import Simulator
from stochastic_matching.common import class_converter


def qs_core_maker(choicer, selector, forbidden_edges=None, threshold=None):
    """
    Parameters
    ----------
    choicer: callable
        Jitted function that extracts the edges that can be selected for a new arrival.
        Input signature is (neighbors list, arriving node, queue_size).
    selector: callable
        Jitted function that extracts one choice amongst several..
        Input signature is (choices, queue_size)
    forbidden_edges: :class:`list`, optional
        Edges that are disabled.
    threshold: :class:`int`
        Queue size above which forbidden edges become available again.

    Returns
    -------
    callable
        A jitted function that will be the core engine of a greedy policy based on the sizes of queues.
    """
    if forbidden_edges is not None:
        forbidden_edges = np.array(forbidden_edges)

    def core_simulator(prob, alias, number_events, seed,
                       neighbors, queue_size,
                       trafic, queue_log, steps_done, threshold=threshold):

        if forbidden_edges is not None:
            forbid = {k: True for k in forbidden_edges}

        # Retrieve number of nodes and max_queue
        n, max_queue = queue_log.shape

        # Place threshold out of range if not defined.
        if threshold is None:
            threshold = max_queue + 1
            no_threshold = True
        else:
            no_threshold = False


        # Initiate random generator if seed is given
        if seed is not None:
            np.random.seed(seed)

        # Start main loop
        age = 0
        for age in range(number_events):

            # Update queue states
            for j in range(n):
                queue_log[j, queue_size[j]] += 1

            # Draw an arrival
            node = np.random.randint(n)
            if np.random.rand() > prob[node]:
                node = alias[node]

            # Check we do not cross the threshold
            contained = no_threshold or np.max(queue_size) < threshold

            # If the arrival queue is non-empty, no new match is feasible with a greedy policy,
            # so we just update and move on unless queue overflows.
            if queue_size[node] > 0 and contained:
                queue_size[node] += 1
                if queue_size[node] == max_queue:
                    return steps_done + age + 1
            else:
                # Otherwise, we can check for feasible edges
                choices = choicer(neighbors, node, queue_size)

                # Do we restrain the edges?
                if forbidden_edges is not None and contained:
                    choices = [ej for ej in choices if (ej[0] not in forbid)]

                if choices:  # At least one possibility
                    if len(choices) == 1:  # Exactly one -> take it.
                        e, j = choices[0]
                    else:  # More than one -> call the selector
                        e, j = selector(choices, queue_size)
                    trafic[e] += 1  # Add trafic for the selected edge
                    queue_size[j] = queue_size[j] - 1  # Decrease queue for neighbor(s)
                else:  # No choice -> update and move on unless queue overflows.
                    queue_size[node] += 1
        return steps_done + age + 1  # Return the updated number of steps achieved.

    return njit(core_simulator)  # Return the jitted core engine.


class QueueSizeSimulator(Simulator, ABC):
    """
    Class derived from :class:`~stochastic_matching.simulator.generic.Simulator`
    for greedy simulator based on queue sizes.
    """
    name = 'generic_queue_size'

    def __init__(self, model, selector = 'longest', selector_kwargs=None,
                              **kwargs):
        self.selector = class_converter(selector, Selector)
        if selector_kwargs is None:
            selector_kwargs = dict()
        self.selector_kwargs = selector_kwargs
        super(QueueSizeSimulator, self).__init__(model, **kwargs)

    def set_inners(self):
        """
        Incorporate `queue_size` to the inner variables.

        Returns
        -------
        None
        """
        super().set_inners()
        self.inners['queue_size'] = np.zeros(self.model.n, dtype=np.uint32)

    def set_core(self):
        """
        Sets the core engine for a greedy policy.

        Returns
        -------
        None
        """
        choicer = SizeChoicer(self.model).yield_jit()
        selector = self.selector(self.model, **self.selector_kwargs).yield_jit()
        self.core = qs_core_maker(choicer, selector)


@njit
def simple_choicer(neighbors, node, queue_size):
    """
    Parameters
    ----------
    neighbors: :class:`~numba.typed.List`
        Output of :func:`~stochastic_matching.common.graph_neighbors_list`
        (for a simple graph).
    node: :class:`int`
        Starting node (the node that just got an arrival).
    queue_size: :class:`~numpy.ndarray`
        Sizes of the different queues.

    Returns
    -------
    :class:`list`
        The (edge, neighbor) tuples of a given node that can be greedily activated in a simple graph.
    """
    return [ej for ej in neighbors[node] if queue_size[ej[1]] > 0]

@njit
def hyper_choicer(neighbors, node, queue_size):
    """
    Parameters
    ----------
    neighbors: :class:`~numba.typed.List`
        Output of :func:`~stochastic_matching.common.graph_neighbors_list`
        (with hypergraph input).
    node: :class:`int`
        Starting node (the node that just got an arrival).
    queue_size: :class:`~numpy.ndarray`
        Sizes of the different queues.

    Returns
    -------
    :class:`list`
        The (edge, neighbors) tuples of a given node that can be greedily activated in a hypergraph.
    """
    return [ej for ej in neighbors[node] if np.all(queue_size[ej[1]] > 0)]


class SizeChoicer:
    """
    Extract possible choices for a greedy policy based on queue size.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model where the choicer will be applied.

    Examples
    --------

    In a diamond graph with non-empty queues in nodes 3 and 0,
    an arrival at node 2 activates (edge, node) (1, 0) and (4, 3).

    >>> import stochastic_matching as sm
    >>> from stochastic_matching.common import graph_neighbors_list
    >>> diamond = sm.CycleChain()
    >>> choicer = SizeChoicer(diamond).yield_jit()
    >>> choicer(graph_neighbors_list(diamond), 2, np.array([1, 0, 0, 1]))
    [(1, 0), (4, 3)]

    Same thing with the diamond seen as hypergraph.

    >>> diamond.adjacency = None
    >>> choicer = SizeChoicer(diamond).yield_jit()
    >>> choices = choicer(graph_neighbors_list(diamond), 2, np.array([1, 0, 0, 1]))
    >>> [(edge, nodes.astype(int)) for edge, nodes in choices]
    [(1, array([0])), (4, array([3]))]

    In a candy hypergraph with non-empty queues in nodes 0, 3, and 4,
    an arrival at node 2 activates (edge, nodes) (1, [0]) and (6, [3, 4]).

    >>> candy = sm.HyperPaddle()
    >>> choicer = SizeChoicer(candy).yield_jit()
    >>> choices = choicer(graph_neighbors_list(candy), 2, np.array([1, 0, 0, 1, 1, 0, 0]))
    >>> [(e, n.astype(int)) for e, n in choices]
    [(1, array([0])), (6, array([3, 4]))]
    """
    name = None
    def __init__(self, model):
        self.model = model
    def yield_jit(self):
        """
        Returns
        -------
        choicer: callable
            Jitted function that extracts the edges that can be selected for a new arrival.
            Input signature is (neighbors list, arriving node, queue_size).
        """
        if self.model.adjacency is not None:
            return simple_choicer
        else:
            return hyper_choicer


class Selector:
    """
    Abstract class for greedy edge selection based on queue sizes.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model where the choicer will be applied.
    """
    name = None

    def __init__(self, model):
        self.model = model

    def yield_selector(self):
        raise NotImplementedError

    def yield_jit(self):
        return njit(self.yield_selector())


class RandomNodeSelector(Selector):
    """
    Selects a feasible edge at random.
    """
    name = 'random_node'

    def yield_selector(self):
        def random_node_selector(choices, queue_size):
            return choices[np.random.randint(len(choices))]
        return random_node_selector


class RandomNodeSimulator(QueueSizeSimulator):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.size_based.QueueSizeSimulator`.
    When multiple choices are possible, one is chosen uniformly at random.

    Parameters
    ----------

    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    **kwargs
        Keyword arguments.


    Examples
    --------

    Let start with a working triangle. One can notice the results are the same for all greedy simulator because
    there are no multiple choices in a triangle (always one non-empty queue at most under a greedy policy).

    >>> import stochastic_matching as sm
    >>> triangle = sm.Cycle(rates=[3, 4, 5])
    >>> sim = RandomNodeSimulator(triangle, number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint64),
     'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
                         [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
                         [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint64),
     'steps_done': 1000}

     Sanity check: results are unchanged if the graph is treated as hypergraph.

    >>> triangle.adjacency = None
    >>> sim = RandomNodeSimulator(triangle, number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint64),
     'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
                         [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
                         [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint64),
     'steps_done': 1000}

    A ill diamond graph (simulation ends before completion due to drift).

    >>> sim = RandomNodeSimulator(sm.CycleChain(rates='uniform'), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([ 7, 10,  1,  4,  7], dtype=uint64),
    'queue_log': array([[22, 13,  7,  7,  5, 15,  4,  0,  0,  0],
           [73,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [69,  3,  1,  0,  0,  0,  0,  0,  0,  0],
           [13, 10, 11,  4,  4,  4,  2,  9, 11,  5]], dtype=uint64),
    'steps_done': 73}

    >>> sim.compute_flow()
    array([0.38356164, 0.54794521, 0.05479452, 0.21917808, 0.38356164])

    A working candy (but candies are not good for greedy policies).

    >>> sim = RandomNodeSimulator(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([26, 21,  4, 25, 34, 10, 16], dtype=uint64),
    'queue_log': array([[ 85,  37,  36,  41,  22,  29,  46,  17,   1,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [275,  32,   7,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [313,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 10,   1,   7,   9,   3,   3,  13,   7,  34,  11,   2,   8,  13,
             18,  12,  56,  27,   8,  24,  25,   7,   8,   3,   1,   4],
           [168,  48,  35,  39,  20,   4,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [278,  25,   7,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [268,  23,  16,   6,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint64),
    'steps_done': 314}

    Note that you can reset the simulator before starting another run.

    >>> sim.reset()
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([26, 21,  4, 25, 34, 10, 16], dtype=uint64),
    'queue_log': array([[ 85,  37,  36,  41,  22,  29,  46,  17,   1,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [275,  32,   7,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [313,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 10,   1,   7,   9,   3,   3,  13,   7,  34,  11,   2,   8,  13,
             18,  12,  56,  27,   8,  24,  25,   7,   8,   3,   1,   4],
           [168,  48,  35,  39,  20,   4,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [278,  25,   7,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [268,  23,  16,   6,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint64),
    'steps_done': 314}

    You can display the distribution of queue sizes as a ccdf:

    >>> sim.show_ccdf() # doctest: +ELLIPSIS
    <Figure size ...x... with 1 Axes>
    """
    name = 'random_node'

    def __init__(self, model, **kwargs):
        super(RandomNodeSimulator, self).__init__(model, selector='random_node', **kwargs)


class LongestSelector(Selector):
    """
    Selects feasible edge with longest queue.
    """
    name = "longest"

    def yield_selector(self):
        """
        Returns
        -------
        callable
            A longest selector function adapted to the model.
        """
        if self.model.adjacency is not None:
            def longest_selector(choices, queue_size):
                i = 0
                q = queue_size[choices[0][1]]
                for j in range(1, len(choices)):
                    qt = queue_size[choices[j][1]]
                    if qt > q:
                        q = qt
                        i = j
                return choices[i]
        else:
            def longest_selector(choices, queue_size):
                i = 0
                q = np.sum(queue_size[choices[0][1]])
                for j in range(1, len(choices)):
                    qt = np.sum(queue_size[choices[j][1]])
                    if qt > q:
                        q = qt
                        i = j
                return choices[i]
        return longest_selector


class LongestSimulator(QueueSizeSimulator):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.size_based.QueueSizeSimulator`.
    When multiple choices are possible, the longest queue (or sum of queues for hyperedges) is chosen.

    Parameters
    ----------

    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    **kwargs
        Keyword arguments.

    Examples
    --------

    Let start with a working triangle. Not that the results are the same for all greedy simulator because
    there are no decision in a triangle (always at most one non-empty queue under a greedy policy).

    >>> import stochastic_matching as sm
    >>> sim = LongestSimulator(sm.Cycle(rates=[3, 4, 5]), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint64),
    'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
       [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
       [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint64),
    'steps_done': 1000}

    A non stabilizable diamond (simulation ends before completion due to drift).

    >>> sim = LongestSimulator(sm.CycleChain(rates='uniform'), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([38, 38,  7, 37, 40], dtype=uint64),
    'queue_log': array([[127,  74,  28,  37,  21,  32,  16,   1,   2,   1],
           [327,   8,   3,   1,   0,   0,   0,   0,   0,   0],
           [322,  12,   4,   1,   0,   0,   0,   0,   0,   0],
           [ 91,  80,  47,  37,  37,  23,  11,   3,   5,   5]], dtype=uint64),
    'steps_done': 339}

    A stabilizable candy (but candies are not good for greedy policies).

    >>> sim = LongestSimulator(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([24, 17,  2, 23, 33, 12, 13], dtype=uint64),
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
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint64),
    'steps_done': 292}
    """
    name = "longest_queue"

    def __init__(self, model, **kwargs):
        super(LongestSimulator, self).__init__(model, selector='longest', **kwargs)


class RandomItemSelector(Selector):
    name = 'random_item'

    def yield_selector(self):
        if self.model.adjacency is not None:
            def random_item_selector(choices, queue_size):
                total = 0
                for i in range(len(choices)):
                    total += queue_size[choices[i][1]]
                target = total * np.random.rand()
                i = 0
                frontier = queue_size[choices[0][1]]
                while target > frontier:
                    i += 1
                    target -= frontier
                    frontier = queue_size[choices[i][1]]
                return choices[i]
        else:
            def random_item_selector(choices, queue_size):
                cumsizes = np.cumsum(np.array([np.sum(queue_size[choices[i][1]]) for i in range(len(choices))]))
                target = cumsizes[-1] * np.random.rand()
                i = 0
                while target > cumsizes[i]:
                    i += 1
                return choices[i]
        return random_item_selector


class RandomItemSimulator(QueueSizeSimulator):
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

    Let start with a working triangle. One can notice the results are the same for all greedy simulator because
    there are no multiple choices in a triangle (always one non-empty queue at most under a greedy policy).

    >>> import stochastic_matching as sm
    >>> sim = RandomItemSimulator(sm.Cycle(rates=[3, 4, 5]), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint64),
    'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
       [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
       [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint64),
    'steps_done': 1000}

    A ill braess graph (simulation ends before completion due to drift).

    >>> sim = RandomItemSimulator(sm.CycleChain(rates='uniform'), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([12, 11,  4,  8, 10], dtype=uint64),
    'queue_log': array([[39, 13, 10,  6,  3,  8, 14,  8,  3,  1],
           [96,  5,  3,  1,  0,  0,  0,  0,  0,  0],
           [97,  7,  1,  0,  0,  0,  0,  0,  0,  0],
           [41, 18, 13, 13,  8,  5,  1,  2,  3,  1]], dtype=uint64),
    'steps_done': 105}

    A working candy (but candies are not good for greedy policies).

    >>> sim = RandomItemSimulator(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([83, 62, 36, 58, 75, 48, 74], dtype=uint64),
    'queue_log': array([[537, 135,  65,  62,  34,  20,  25,  30,  48,  12,   4,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [792, 130,  28,  14,   8,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [861,  71,  19,  15,   5,   1,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 10,   1,   7,   9,   3,  31,  65,  70,  46,  56,  60,  82,  59,
             49,  54,  60,  61,  42, 100,  44,  28,  10,  15,   9,   1],
           [711, 127,  77,  34,  19,   4,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [800,  97,  50,  22,   3,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [732, 125,  74,  25,  11,   3,   2,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint64),
    'steps_done': 972}
    """
    name = 'random_item'

    def __init__(self, model, **kwargs):
        super(RandomItemSimulator, self).__init__(model, selector='random_item', **kwargs)


class FilteringGreedy(QueueSizeSimulator):
    """
    Longest queue simulator where some edges can be forbidden unless some threshold on queue size is reached.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    forbidden_edges: :class:`list` or :class:`~numpy.ndarray`, optional
        Egdes that should not be used.
    weights: :class:`~numpy.ndarray`, optional
        Target rewards on edges. If weights are given, the forbidden edges are computed to match the target
        (overrides forbidden_edges argument).
    threshold: :class:`int`, optional
        Limit on queue size to apply edge interdiction (enforce stability on injective-only vertices).
    **kwargs
        Keyword arguments.

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> diamond = sm.CycleChain(rates=[1, 2, 2, 1])
    >>> diamond.run('filtering', forbidden_edges=[0, 4], seed=42,
    ...                            threshold=100, number_events=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([0.   , 0.954, 0.966, 0.954, 0.   ])

    Same result can be achieved by putting low weights on 0 and 4.

    >>> diamond.run('filtering', weights=[1, 2, 2, 2, 1], seed=42,
    ...                            threshold=100, number_events=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([0.   , 0.954, 0.966, 0.954, 0.   ])

    To compare with the priority-based pure greedy version:

    >>> diamond.run('priority', weights=[1, 2, 2, 2, 1], number_events=1000, max_queue=1000, seed=42)
    True
    >>> diamond.simulation
    array([0.444, 0.63 , 0.966, 0.63 , 0.324])

    Another example with other rates.

    >>> diamond.rates=[4, 5, 2, 1]

    Optimize with the first and last edges that provide less reward.

    >>> diamond.run('filtering', weights=[1, 2, 2, 2, 1], seed=42,
    ...                            threshold=100, number_events=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([3.264, 0.888, 0.948, 0.84 , 0.   ])

    Increase the reward on the first edge.

    >>> diamond.run('filtering', weights=[4, 2, 2, 2, 1], seed=42,
    ...                            threshold=100, number_events=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([4.152, 0.   , 0.996, 0.   , 0.84 ])

    On bijective graphs, no edge is forbidden whatever the weights.

    >>> paw = sm.Tadpole()
    >>> paw.run('filtering', weights=[6, 3, 1, 2], seed=42,
    ...                            threshold=100, number_events=1000, max_queue=1000)
    True
    >>> paw.simulation
    array([1.048, 1.056, 1.016, 0.88 ])
    """
    name = 'filtering'

    def __init__(self, model, forbidden_edges=None, threshold=None, weights=None, **kwargs):
        super(FilteringGreedy, self).__init__(model, **kwargs)
        if weights is not None:
            weights = np.array(weights)
            flow = model.optimize_rates(weights)
            forbidden_edges = [i for i in range(model.m) if flow[i]==0]
            if len(forbidden_edges) == 0:
                forbidden_edges = None
        choicer = SizeChoicer(self.model).yield_jit()
        selector = LongestSelector(self.model).yield_jit()
        self.core = qs_core_maker(choicer, selector, forbidden_edges=forbidden_edges, threshold=threshold)


@njit
def priority_with_weights(choices, queue_size, weights):
    i = np.argmax(np.array([weights[e] for (e, j) in choices]))
    return choices[i]


@njit
def priority_with_weights_and_threshold(choices, queue_size, weights,
                                        threshold, counterweights):
    if np.max(queue_size) >= threshold:
        i = np.argmax(np.array([counterweights[e] for (e, j) in choices]))
    else:
        i = np.argmax(np.array([weights[e] for (e, j) in choices]))
    return choices[i]


class PrioritySelector(Selector):
    """
    Selects feasible edge based on priorities.
    """
    name = 'priority'

    def __init__(self, model, weights, threshold=None, counterweights=None):
        weights = np.array(weights)
        self.weights = weights
        self.threshold = threshold
        if threshold is not None:
            if counterweights is None:
                counterweights = -weights
            else:
                counterweights = np.array(counterweights)
            self.counterweights = counterweights
        else:
            self.counterweights = None
        super(PrioritySelector, self).__init__(model)

    def yield_selector(self):
        pass

    def yield_jit(self):
        weights = self.weights.copy()
        if self.threshold is None:
            def weighted_choice(choices, queue_size):
                return priority_with_weights(choices, queue_size, weights)
        else:
            counterweights = self.counterweights.copy()
            threshold = self.threshold

            def weighted_choice(choices, queue_size):
                return priority_with_weights_and_threshold(choices, queue_size,
                                                           weights, threshold, counterweights)
        return njit(weighted_choice)


class PrioritySimulator(QueueSizeSimulator):
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
    ...                          number_events=10000, seed=42)
    True

    These priorities are efficient at stabilizing the policy while avoiding edge 3.

    >>> fish.simulation
    array([2.925 , 1.0404, 0.9522, 0.    , 0.9504, 2.0808, 1.0044])

    The last node is the pseudo-instable node.

    >>> fish.simulator.compute_average_queues()[-1]
    38.3411
    >>> import numpy as np
    >>> np.round(np.mean(fish.simulator.compute_average_queues()[:-1]), decimals=2)
    0.75

    Choosing proper counter-weights is important.

    >>> fish.run('priority', weights=[0, 2, 2, 0, 1, 1, 0],
    ...                          threshold=50,
    ...                          number_events=10000, seed=42)
    True
    >>> fish.simulation
    array([2.9232, 1.0422, 0.9504, 0.216 , 0.7344, 1.8666, 1.2186])
    >>> fish.simulator.compute_average_queues()[-1]
    38.5966
    """
    name = 'priority'

    def __init__(self, model, weights=None, threshold=None, counterweights=None, **kwargs):
        super(PrioritySimulator, self).__init__(model,
                                                     selector='priority',
                                                     selector_kwargs = {'weights': weights,
                                                                       'threshold': threshold,
                                                                       'counterweights': counterweights},
                                                     **kwargs)
