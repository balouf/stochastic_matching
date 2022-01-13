import numpy as np
from numba import njit

import os
if os.environ.get("NUMBA_DISABLE_JIT") == "1":
    List = lambda x: x
else:
    from numba.typed import List

from stochastic_matching.graphs.classes import SimpleGraph, HyperGraph


@njit
def set_seed(value):
    """
    Change the random generator seed inside numba jitted scope.

    Parameters
    ----------
    value: :class:`int`
        Seed.

    Returns
    -------
    None
    """
    np.random.seed(value)


def create_prob_alias(mu):
    """
    Prepare vector to draw a distribution with the alias method.

    Parameters
    ----------
    mu: :class:`list` or :class:`~numpy.ndarray`
        Arrival intensities.

    Returns
    -------
    prob: :class:`~numpy.ndarray`
        Probabilities to stay in the drawn bucket
    alias: :class:`~numpy.ndarray`
        Redirection array

    Examples
    --------

    >>> probas, aliases = create_prob_alias([2 ,2, 3, 1])
    >>> probas
    array([1. , 1. , 1. , 0.5])
    >>> aliases
    array([0, 0, 0, 2])
    """
    cmu = np.array(mu)
    n = len(cmu)
    alias = np.zeros(n, dtype=int)
    prob = np.zeros(n)
    # noinspection PyUnresolvedReferences
    normalized_intensities = cmu * n / np.sum(cmu)
    small = [i for i in range(n) if normalized_intensities[i] < 1]
    large = [i for i in range(n) if normalized_intensities[i] >= 1]
    while small and large:
        l, g = small.pop(), large.pop()
        prob[l], alias[l] = normalized_intensities[l], g
        normalized_intensities[g] += normalized_intensities[l] - 1
        if normalized_intensities[g] < 1:
            small.append(g)
        else:
            large.append(g)
    for i in large + small:
        prob[i] = 1
    return prob, alias


def graph_neighbors_list(graph):
    """
    Extract Numba-compatible neighboring structures from a :class:`~stochastic_matching.graphs.graph.GenericGraph`.

    Parameters
    ----------
    graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graphs.classes.HyperGraph`
        Graph to transform.

    Returns
    -------
    :class:`~numba.typed.List`
        List of neighbors for each node. For one given node yields a list of tuples where the first element is the edge
        and the second element the neighbor (for `SimpleGraph`) / array of neighbors (for `HyperGraph`).

    Examples
    --------

    Consider a Braess graph.

    >>> from stochastic_matching import bicycle_graph, hyper_paddle
    >>> braess = bicycle_graph()
    >>> braess.incidence.toarray().astype(int)
    array([[1, 1, 0, 0, 0],
           [1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1],
           [0, 0, 0, 1, 1]])

    Node 0 is connected with edge 0 to node 1 and with edge 1 to node 2.

    Node 1 is connected with edge 0 to node 0, with edge 2 to node 2, and with edge 3 to node 3.

    Node 2 is connected with edge 1 to node 0, with edge 2 to node 1, and with edge 4 to node 3.

    Node 3 is connected with edge 3 to node 1 and with edge 4 to node 2.

    This is exactly what `graph_neighbors_list` outputs, in a numba-compatible way.

    >>> graph_neighbors_list(braess) # doctest: +NORMALIZE_WHITESPACE
    [[(0, 1), (1, 2)],
    [(0, 0), (2, 2), (3, 3)],
    [(1, 0), (2, 1), (4, 3)],
    [(3, 1), (4, 2)]]

    With hypergraph notation, the second term of the tuples is an array instead of an integer.

    >>> g = graph_neighbors_list(braess.to_hypergraph())
    >>> [ [(e, a.astype(int)) for e, a in n] for n in g ] # doctest: +NORMALIZE_WHITESPACE
    [[(0, array([1])), (1, array([2]))],
    [(0, array([0])), (2, array([2])), (3, array([3]))],
    [(1, array([0])), (2, array([1])), (4, array([3]))],
    [(3, array([1])), (4, array([2]))]]


    Having arrays is only useful for true hypergraphs. For instance, in the candy hypergraph, edge 6 links
    nodes 2, 3, and 4 together.

    >>> candy = hyper_paddle()
    >>> candy.incidence.toarray().astype(int)
    array([[1, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1, 0, 1],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 1, 0]])

    This shows in the output of the function.

    >>> g = graph_neighbors_list(candy)
    >>> [ [(e, a.astype(int)) for e, a in n] for n in g ] # doctest: +NORMALIZE_WHITESPACE
    [[(0, array([1])), (1, array([2]))],
    [(0, array([0])), (2, array([2]))],
    [(1, array([0])), (2, array([1])), (6, array([3, 4]))],
    [(6, array([2, 4]))],
    [(3, array([5])), (4, array([6])), (6, array([2, 3]))],
    [(3, array([4])), (5, array([6]))],
    [(4, array([4])), (5, array([5]))]]

    Trying the function on something that is not SimpleGraph or HyperGraph raises an error.

    >>> graph_neighbors_list(candy.incidence)
    Traceback (most recent call last):
    ...
    TypeError: graph must be of type SimpleGraph of HyperGraph.
    """
    if not (isinstance(graph, SimpleGraph) or isinstance(graph, HyperGraph)):
        raise TypeError("graph must be of type SimpleGraph of HyperGraph.")
    edges = [graph.incidence.indices[graph.incidence.indptr[i]:graph.incidence.indptr[i + 1]] for i in range(graph.n)]
    if isinstance(graph, SimpleGraph):
        neighbors = [[[k for k in graph.co_incidence.indices[graph.co_incidence.indptr[e]:graph.co_incidence.indptr[e + 1]] if k != i][0]
                      for e in edges[i]] for i in range(graph.n)]
    elif isinstance(graph, HyperGraph):
        neighbors = [
            [np.array([k for k in graph.co_incidence.indices[graph.co_incidence.indptr[e]:graph.co_incidence.indptr[e + 1]] if k != i], dtype=np.int32)
             for e in edges[i]] for i in range(graph.n)]
    return List([List([(e, v) for e, v in zip(edges[i], neighbors[i])]) for i in range(graph.n)])
