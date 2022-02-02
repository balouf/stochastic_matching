from abc import ABC

import numpy as np
from numba import njit

from stochastic_matching.simulator.generic import Simulator
from stochastic_matching.common import graph_neighbors_list


def qstate_core_maker(model, simple_selector, hyper_selector):
    """
    Parameters
    ----------
    graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graphs.classes.HyperGraph`
        The graph for which the simulation is intended.
    simple_selector: callable
        Jitted function that extracts one choice amongst several for simple graphs.
        Input signature is (neighbors list, queue_size).
    hyper_selector: callable
        Jitted function that extracts one choice amongst several for hypergraphs.
        Input signature is (neighbors list, queue_size)

    Returns
    -------
    callable
        A jitted function that will be the core engine of a greedy policy based on the states of queues.
    """
    # prepare the correct functions depending on graph type.
    if model.adjacency is not None:
        choicer = simple_state_choicer
        selector = simple_selector
    elif model.adjacency is None:
        choicer = hyper_state_choicer
        selector = hyper_selector

    def core_simulator(prob, alias, number_events, seed,
                       neighbors,
                       queue_start, queue_end, items,
                       trafic, queue_log, steps_done):

        # Retrieve number of nodes and max_queue
        n, max_queue = queue_log.shape

        # Initiate random generator if seed is given
        if seed is not None:
            np.random.seed(seed)

        # Start main loop
        age = 0
        for age in range(number_events):

            # Update queue logs
            for j in range(n):
                queue_log[j, queue_end[j] - queue_start[j]] += 1

            # Draw an arrival
            node = np.random.randint(n)
            if np.random.rand() > prob[node]:
                node = alias[node]

            # node=int(node)

            # If the arrival queue is non-empty, no new match is feasible with a greedy policy,
            # so we just update and move on unless queue overflows.
            if queue_end[node] > queue_start[node]:
                items[node, queue_end[node] % max_queue] = age
                queue_end[node] += 1
                if (queue_end[node] - queue_start[node]) == max_queue:
                    return steps_done + age + 1

            else:
                # Otherwise, we can check for feasible edges
                choices = choicer(neighbors, node, queue_start, queue_end)
                if choices:  # At least one possibility
                    if len(choices) == 1:  # Exactly one -> take it.
                        e, j = choices[0]
                    else:  # More than one -> call the selector
                        e, j = selector(choices, max_queue, queue_start, queue_end, items)
                    trafic[e] += 1  # Add trafic for the selected edge
                    queue_start[j] = queue_start[j] + 1  # "Pop" oldest item(s) from selected edge.
                else:  # No choice -> update and move on unless queue overflows.
                    items[node, queue_end[node] % max_queue] = age
                    queue_end[node] += 1
                    if (queue_end[node] - queue_start[node]) == max_queue:
                        return steps_done + age + 1
        return steps_done + age + 1  # Return the updated number of steps achieved.

    return njit(core_simulator, cache=True)  # Return the jitted core engine.


class QueueStateSimulator(Simulator, ABC):
    """
    Abstract class derived from :class:`~stochastic_matching.simulator.generic.Simulator`
    for greedy simulator based on the states of queues (including age of items).
    """

    def set_inners(self):
        """
        Incorporate `queue_start`, `queue_end`, and `items` to the inner variables.

        Returns
        -------
        None
        """
        super().set_inners()
        self.inners['queue_start'] = np.zeros(self.model.n, dtype=np.uint32)
        self.inners['queue_end'] = np.zeros(self.model.n, dtype=np.uint32)
        self.inners['items'] = np.zeros((self.model.n, self.max_queue), dtype=np.uint32)

    def set_core_from_selector(self, simple_selector, hyper_selector):
        """
        Sets the core engine for a greedy policy.

        Parameters
        ----------
        simple_selector: callable
            Jitted selector function for simple graphs.
        hyper_selector: callable
            Jitted selector function for hypergraphs.

        Returns
        -------
        None
        """
        self.core = qstate_core_maker(self.model, simple_selector, hyper_selector)


@njit
def simple_state_choicer(neighbors, node, queue_start, queue_end):
    """
    Parameters
    ----------
    neighbors: :class:`~numba.typed.List`
        Output of :func:`~stochastic_matching.common.graph_neighbors_list`.
    node: :class:`int`
        Starting node (the node that just got an arrival).
    queue_start: :class:`~numpy.ndarray`
        Starting queue pointers.
    queue_end: :class:`~numpy.ndarray`
        Ending queue pointers.

    Returns
    -------
    :class:`list`
        The (edge, neighbor) tuples of a given node that can be greedily activated in a simple graph.

    Examples
    --------

    In a diamond with non-empty queues in nodes 3 and 0,
    an arrival at node 2 activates (edge, node) (1, 0) and (4, 3).

    >>> import stochastic_matching as sm
    >>> simple_state_choicer(graph_neighbors_list(sm.CycleChain()), 2,
    ...                      np.array([10, 14, 7, 8]), np.array([11, 14, 7, 9]))
    [(1, 0), (4, 3)]
    """
    return [ej for ej in neighbors[node] if (queue_end[ej[1]] - queue_start[ej[1]]) > 0]


@njit
def hyper_state_choicer(neighbors, node, queue_start, queue_end):
    """
    Parameters
    ----------
    neighbors: :class:`~numba.typed.List`
        Output of :func:`~stochastic_matching.common.graph_neighbors_list`.
    node: :class:`int`
        Starting node (the node that just got an arrival).
    queue_start: :class:`~numpy.ndarray`
        Starting queue pointers.
    queue_end: :class:`~numpy.ndarray`
        Ending queue pointers.

    Returns
    -------
    :class:`list`
        The (edge, neighbors) tuples of a given node that can be greedily activated in a hypergraph.

    Examples
    --------

    In a candy hypergraph with non-empty queues in nodes 0, 3, and 4,
    an arrival at node 2 activates (edge, nodes) (1, [0]) and (6, [3, 4]).

    >>> import stochastic_matching as sm
    >>> choices = hyper_state_choicer(graph_neighbors_list(sm.HyperPaddle()), 2,
    ...                     np.array([21, 10, 7, 4, 3, 2, 50]), np.array([22, 10, 7, 5, 4, 2, 50]))
    >>> [(e, n.astype(int)) for e, n in choices]
    [(1, array([0])), (6, array([3, 4]))]
    """
    # noinspection PyUnresolvedReferences
    return [ej for ej in neighbors[node] if np.all((queue_end[ej[1]] - queue_start[ej[1]]) > 0)]


@njit
def fcfm_selector(choices, max_queue, queue_start, queue_end, items):
    """
    Select the edge with oldest item.

    Parameters
    ----------
    choices: :class:`list`
        (edge, neighbors) list to choose from.
    max_queue: :class:`int`
        Max queue size.
    queue_start: :class:`~numpy.ndarray`
        Starting queue pointers.
    queue_end: :class:`~numpy.ndarray`
        Ending queue pointers.
    items: :class:`~numpy.ndarray`
        Ages of items (current items are accessed through queue pointers).

    Returns
    -------
    :class:`tuple`
        Selected (edge, neighbors)

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> start = np.array([1, 0, 0, 2])
    >>> end = np.array([2, 0, 0, 4])
    >>> items_list = np.array([[1, 5, 6, 0, 0],
    ...                   [0, 0, 0, 0, 0],
    ...                   [0, 0, 0, 0, 0],
    ...                   [2, 3, 4, 7, 8],])
    >>> fcfm_selector(simple_state_choicer(graph_neighbors_list(sm.CycleChain()), 2,
    ...                                             start, end),
    ...                        5, start, end, items_list)
    (4, 3)
    """
    i = 0
    target = choices[0][1]
    age = items[target, (queue_start[target]) % max_queue]
    for j in range(1, len(choices)):
        target = choices[j][1]
        new_age = items[target, (queue_start[target]) % max_queue]
        if new_age < age:
            age = new_age
            i = j
    return choices[i]


@njit
def fcfm_hyper_selector(choices, max_queue, queue_start, queue_end, items):
    """
    Select the edge with oldest item.

    Parameters
    ----------
    choices: :class:`list`
        (edge, neighbors) list to choose from.
    max_queue: :class:`int`
        Max queue size.
    queue_start: :class:`~numpy.ndarray`
        Starting queue pointers.
    queue_end: :class:`~numpy.ndarray`
        Ending queue pointers.
    items: :class:`~numpy.ndarray`
        Ages of items (current items are accessed through queue pointers).

    Returns
    -------
    :class:`tuple`
        Selected (edge, neighbors)

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> start = np.array([0, 0, 0, 0, 0, 0, 0])
    >>> end = np.array([1, 0, 0, 1, 1, 0, 0])
    >>> items_list = np.array([[2, 0], [0, 0], [0, 0], [1, 0], [3, 0], [0, 0], [0, 0]])
    >>> e, n = fcfm_hyper_selector(hyper_state_choicer(graph_neighbors_list(sm.HyperPaddle()), 2,
    ...                                             start, end),
    ...                        2, start, end, items_list)
    >>> e
    6
    >>> n.astype(int)
    array([3, 4])
    >>> items_list = np.array([[1, 0], [0, 0], [0, 0], [2, 0], [3, 0], [0, 0], [0, 0]])
    >>> e, n = fcfm_hyper_selector(hyper_state_choicer(graph_neighbors_list(sm.HyperPaddle()), 2,
    ...                                             start, end),
    ...                        2, start, end, items_list)
    >>> e
    1
    >>> n.astype(int)
    array([0])
    """
    i = 0
    targets = choices[0][1]
    empty = True
    if len(targets) > 0:
        age = np.min(np.array([items[t, queue_start[t] % max_queue] for t in targets]))
        empty = False
    for j in range(1, len(choices)):
        targets = choices[j][1]
        if len(targets) > 0:
            new_age = np.min(np.array([items[t, queue_start[t] % max_queue] for t in targets]))
            if empty or (new_age < age):
                age = new_age
                i = j
                empty = False
    return choices[i]


class FCFM(QueueStateSimulator):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.age_based.QueueStateSimulator`.
    When multiple choices are possible, the oldest item is chosen.

    Examples
    --------

    Let start with a working triangle. One can notice the results are the same for all greedy simulator because
    there are no multiple choices in a triangle (always one non-empty queue at most under a greedy policy).

    >>> import stochastic_matching as sm
    >>> sim = sm.FCFM(sm.Cycle(rates=[3, 4, 5]), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint64),
    'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
       [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
       [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint64),
    'steps_done': 1000}

    Unstable diamond (simulation ends before completion due to drift).

    >>> sim = FCFM(sm.CycleChain(rates=[1, 1, 1, 1]), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([34, 42,  7, 41, 36], dtype=uint64),
    'queue_log': array([[127,  70,  22,  26,  29,  12,  23,  15,  10,   5],
           [327,   8,   3,   1,   0,   0,   0,   0,   0,   0],
           [322,  12,   4,   1,   0,   0,   0,   0,   0,   0],
           [106,  80,  65,  28,  31,  15,   4,   2,   6,   2]], dtype=uint64),
    'steps_done': 339}

    A stable candy (but candies are not good for greedy policies).

    >>> sim = FCFM(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]),
    ...            number_events=1000, seed=42, max_queue=25)
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

    name = 'fcfm'
    """
    String that can be use to refer to that simulator.
    """

    def set_core(self):
        """
        Build the core engine to choose the oldest item.

        Returns
        -------
        None
        """
        self.set_core_from_selector(fcfm_selector, fcfm_hyper_selector)
