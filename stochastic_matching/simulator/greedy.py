import numpy as np
from numba import njit

from stochastic_matching.graphs.classes import SimpleGraph, HyperGraph
from stochastic_matching.simulator.common import graph_neighbors_list, set_seed


@njit
def simple_choicer(neighbors, node, queue_size):
    """
    Parameters
    ----------
    neighbors: :class:`~numba.typed.List`
        Output of :func:`~stochastic_matching.simulator.common.graph_neighbors_list`
        (with :class:`~stochastic_matching.graphs.classes.SimpleGraph` input).
    node: :class:`int`
        Starting node (the node that just got an arrival).
    queue_size: :class:`~numpy.ndarray`
        Sizes of the different queues.

    Returns
    -------
    :class:`list`
        The (edge, neighbor) tuples of a given node that can be greedily activated in a simple graph.

    Examples
    --------

    In a Braess graph with non-empty queues in nodes 3 and 0,
    an arrival at node 2 activates (edge, node) (1, 0) and (4, 3).

    >>> from stochastic_matching import bicycle_graph
    >>> simple_choicer(graph_neighbors_list(bicycle_graph()), 2, np.array([1, 0, 0, 1]))
    [(1, 0), (4, 3)]
    """
    return [ej for ej in neighbors[node] if queue_size[ej[1]] > 0]


@njit
def hyper_choicer(neighbors, node, queue_size):
    """
    Parameters
    ----------
    neighbors: :class:`~numba.typed.List`
        Output of :func:`~stochastic_matching.simulator.common.graph_neighbors_list`
        (with :class:`~stochastic_matching.graphs.classes.HyperGraph` input).
    node: :class:`int`
        Starting node (the node that just got an arrival).
    queue_size: :class:`~numpy.ndarray`
        Sizes of the different queues.

    Returns
    -------
    :class:`list`
        The (edge, neighbors) tuples of a given node that can be greedily activated in a hypergraph.

    Examples
    --------

    In a candy hypergraph with non-empty queues in nodes 0, 3, and 4,
    an arrival at node 2 activates (edge, nodes) (1, [0]) and (6, [3, 4]).

    >>> from stochastic_matching import hyper_paddle
    >>> choices = hyper_choicer(graph_neighbors_list(hyper_paddle()), 2, np.array([1, 0, 0, 1, 1, 0, 0]))
    >>> [(e, n.astype(int)) for e, n in choices]
    [(1, array([0])), (6, array([3, 4]))]
    """
    # noinspection PyUnresolvedReferences
    return [ej for ej in neighbors[node] if np.all(queue_size[ej[1]] > 0)]


def qs_core_maker(graph, simple_selector, hyper_selector):
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
        A jitted function that will be the core engine of a greedy policy based on the sizes of queues.
    """
    # prepare the correct functions depending on graph type.
    if isinstance(graph, SimpleGraph):
        choicer = simple_choicer
        selector = simple_selector
    elif isinstance(graph, HyperGraph):
        choicer = hyper_choicer
        selector = hyper_selector

    def core_simulator(prob, alias, number_events, seed,
                       neighbors, queue_size,
                       trafic, queue_log, steps_done):

        # Retrieve number of nodes and max_queue
        n, max_queue = queue_log.shape

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

            # If the arrival queue is non-empty, no new match is feasible with a greedy policy,
            # so we just update and move on unless queue overflows.
            if queue_size[node] > 0:
                queue_size[node] += 1
                if queue_size[node] == max_queue:
                    return steps_done + age + 1
            else:
                # Otherwise, we can check for feasible edges
                choices = choicer(neighbors, node, queue_size)
                if choices:  # At least one possibility
                    if len(choices) == 1:  # Exactly one -> take it.
                        e, j = choices[0]
                    else:  # More than one -> call the selector
                        e, j = selector(choices, queue_size)
                    trafic[e] += 1  # Add trafic for the selected edge
                    queue_size[j] = queue_size[j] - 1  # Decrease queue for neighbor(s)
                else:  # No choice -> update and move on unless queue overflows.
                    queue_size[node] = 1
        return steps_done + age + 1  # Return the updated number of steps achieved.

    return njit(core_simulator, cache=True)  # Return the jitted core engine.


@njit
def random_node_selector(choices, queue_size):
    """
    Select a choice at random.

    Parameters
    ----------
    choices: :class:`list`
        (edge, neighbor(s)) list to choose from
    queue_size: :class:`~numpy.ndarray`
        Sizes of queue (not used for this specific selector).

    Returns
    -------
    :class:`tuple`
        Selected (edge, neighbor(s))

    Examples
    --------
    >>> from stochastic_matching import bicycle_graph, hyper_paddle
    >>> set_seed(42)
    >>> qs = np.array([1, 0, 0, 1])
    >>> random_node_selector(simple_choicer(graph_neighbors_list(bicycle_graph()), 2, qs), qs)
    (1, 0)
    >>> qs = np.array([1, 0, 0, 1, 1, 0, 0])
    >>> e, n = random_node_selector(hyper_choicer(graph_neighbors_list(hyper_paddle()), 2, qs), qs)
    >>> e
    6
    >>> n.astype(int)
    array([3, 4])
    """
    return choices[np.random.randint(len(choices))]


@njit
def longest_queue_selector(choices, queue_size):
    """
    Select a choice with longest queue size.

    Parameters
    ----------
    choices: :class:`list`
        (edge, neighbors) list to choose from.
    queue_size: :class:`~numpy.ndarray`
        Sizes of queue.

    Returns
    -------
    :class:`tuple`
        Selected (edge, neighbors)

    Examples
    --------
    >>> from stochastic_matching import bicycle_graph
    >>> qs = np.array([1, 0, 0, 2])
    >>> longest_queue_selector(simple_choicer(graph_neighbors_list(bicycle_graph()), 2, qs), qs)
    (4, 3)
    """
    i = 0
    q = queue_size[choices[0][1]]
    for j in range(1, len(choices)):
        qt = queue_size[choices[j][1]]
        if qt > q:
            q = qt
            i = j
    return choices[i]


@njit
def longest_sum_queue_selector(choices, queue_size):
    """
    Select a choice with longest sum of queues.

    Parameters
    ----------
    choices: :class:`list`
        (edge, neighbors) list to choose from
    queue_size: :class:`~numpy.ndarray`
        Sizes of queue.

    Returns
    -------
    :class:`tuple`
        Selected (edge, neighbors)

    Examples
    --------
    >>> from stochastic_matching import hyper_paddle
    >>> qs = np.array([3, 0, 0, 2, 2, 0, 0])
    >>> e, n = longest_sum_queue_selector(hyper_choicer(graph_neighbors_list(hyper_paddle()), 2, qs), qs)
    >>> e
    6
    >>> n.astype(int)
    array([3, 4])
    """
    i = 0
    # noinspection PyUnresolvedReferences
    q = np.sum(queue_size[choices[0][1]])
    for j in range(1, len(choices)):
        # noinspection PyUnresolvedReferences
        qt = np.sum(queue_size[choices[j][1]])
        if qt > q:
            q = qt
            i = j
    return choices[i]


@njit
def random_item_selector(choices, queue_size):
    """
    Select proportionally to queue sizes.

    Parameters
    ----------
    choices: :class:`list`
        (edge, neighbors) list to choose from.
    queue_size: :class:`~numpy.ndarray`
        Sizes of queue.

    Returns
    -------
    :class:`tuple`
        Selected (edge, neighbors)

    Examples
    --------
    >>> from stochastic_matching import bicycle_graph
    >>> set_seed(42)
    >>> qs = np.array([1, 0, 0, 2])
    >>> random_item_selector(simple_choicer(graph_neighbors_list(bicycle_graph()), 2, qs), qs)
    (4, 3)
    """
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


@njit
def random_sum_item_selector(choices, queue_size):
    """
    Select proportionally to sum of adjacent queues (e.g. sum of queues of nodes that belong to a given edge).

    Parameters
    ----------
    choices: :class:`list`
        (edge, neighbors) list to choose from
    queue_size: :class:`~numpy.ndarray`
        Sizes of queue.

    Returns
    -------
    :class:`tuple`
        Selected (edge, neighbors)

    Examples
    --------
    >>> from stochastic_matching import hyper_paddle
    >>> qs = np.array([3, 0, 0, 2, 2, 0, 0])
    >>> e, n = random_sum_item_selector(hyper_choicer(graph_neighbors_list(hyper_paddle()), 2, qs), qs)
    >>> e
    6
    >>> n.astype(int)
    array([3, 4])
    """
    # noinspection PyUnresolvedReferences
    cumsizes = np.cumsum(np.array([np.sum(queue_size[choices[i][1]]) for i in range(len(choices))]))
    target = cumsizes[-1] * np.random.rand()
    i = 0
    while target > cumsizes[i]:
        i += 1
    return choices[i]


@njit
def simple_state_choicer(neighbors, node, queue_start, queue_end):
    """
    Parameters
    ----------
    neighbors: :class:`~numba.typed.List`
        Output of :func:`~stochastic_matching.simulator.common.graph_neighbors_list`
        (with :class:`~stochastic_matching.graphs.classes.SimpleGraph` input).
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

    In a Braess graph with non-empty queues in nodes 3 and 0,
    an arrival at node 2 activates (edge, node) (1, 0) and (4, 3).

    >>> from stochastic_matching import bicycle_graph
    >>> simple_state_choicer(graph_neighbors_list(bicycle_graph()), 2,
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
        Output of :func:`~stochastic_matching.simulator.common.graph_neighbors_list`
        (with :class:`~stochastic_matching.graphs.classes.HyperGraph` input).
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

    >>> from stochastic_matching import hyper_paddle
    >>> choices = hyper_state_choicer(graph_neighbors_list(hyper_paddle()), 2,
    ...                     np.array([21, 10, 7, 4, 3, 2, 50]), np.array([22, 10, 7, 5, 4, 2, 50]))
    >>> [(e, n.astype(int)) for e, n in choices]
    [(1, array([0])), (6, array([3, 4]))]
    """
    # noinspection PyUnresolvedReferences
    return [ej for ej in neighbors[node] if np.all((queue_end[ej[1]] - queue_start[ej[1]]) > 0)]


def qstate_core_maker(graph, simple_selector, hyper_selector):
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
    if isinstance(graph, SimpleGraph):
        choicer = simple_state_choicer
        selector = simple_selector
    elif isinstance(graph, HyperGraph):
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
        return steps_done + age + 1  # Return the updated number of steps achieved.

    return njit(core_simulator, cache=True)  # Return the jitted core engine.


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
    >>> from stochastic_matching import bicycle_graph
    >>> start = np.array([1, 0, 0, 2])
    >>> end = np.array([2, 0, 0, 4])
    >>> items_list = np.array([[1, 5, 6, 0, 0],
    ...                   [0, 0, 0, 0, 0],
    ...                   [0, 0, 0, 0, 0],
    ...                   [2, 3, 4, 7, 8],])
    >>> fcfm_selector(simple_state_choicer(graph_neighbors_list(bicycle_graph()), 2,
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
    >>> from stochastic_matching import hyper_paddle
    >>> start = np.array([0, 0, 0, 0, 0, 0, 0])
    >>> end = np.array([1, 0, 0, 1, 1, 0, 0])
    >>> items_list = np.array([[2, 0], [0, 0], [0, 0], [1, 0], [3, 0], [0, 0], [0, 0]])
    >>> e, n = fcfm_hyper_selector(hyper_state_choicer(graph_neighbors_list(hyper_paddle()), 2,
    ...                                             start, end),
    ...                        2, start, end, items_list)
    >>> e
    6
    >>> n.astype(int)
    array([3, 4])
    >>> items_list = np.array([[1, 0], [0, 0], [0, 0], [2, 0], [3, 0], [0, 0], [0, 0]])
    >>> e, n = fcfm_hyper_selector(hyper_state_choicer(graph_neighbors_list(hyper_paddle()), 2,
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
