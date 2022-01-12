import numpy as np
from stochastic_matching.graphs.classes import SimpleGraph, HyperGraph

def pan(cycle=3, tail=1, names=None):
    """
    Parameters
    ----------
    cycle: :class:`int`
        Length of the cycle.
    tail: :class:`int`
        Length of the tail
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A graph with tail of number_events `tail` attached to a cycle of number_events `cycle`.

    Examples
    --------

    A triangle with a one-edge tail:

    >>> simple_pan = pan(cycle=3, tail=1)
    >>> simple_pan.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 1],
           [0, 0, 1, 0]])

    A pentacle:

    >>> pentacle = pan(cycle=5, tail=0)
    >>> pentacle.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 0, 1],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [0, 0, 1, 0, 1],
           [1, 0, 0, 1, 0]])

    A larger pan:

    >>> long_pan = pan(cycle=4, tail=3)
    >>> long_pan.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 0, 1, 0]])
    """
    n = cycle + tail
    adja = np.zeros([n, n], dtype=int)
    for i in range(cycle):
        adja[i, (i + 1) % cycle] = 1
        adja[i, (i - 1) % cycle] = 1
    for i in range(tail):
        adja[cycle + i - 1, cycle + i] = 1
        adja[cycle + i, cycle + i - 1] = 1
    return SimpleGraph(adjacency=adja, names=names)


def bicycle(left_cycle=3, right_cycle=3, common_edges=1, names=None):
    """
    Parameters
    ----------
    left_cycle: :class:`int`
        Size of the first cycle.
    right_cycle: :class:`int`
        Size of the second cycle.
    common_edges: :class:`int`
        Number of common edges
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A graph with with two cycles sharing common edges/nodes.

    Examples
    --------

    A *house* (a square and a triangle with one common edge).

    >>> house = bicycle(left_cycle=4, right_cycle=3, common_edges=1)
    >>> house.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 0],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 1],
           [1, 0, 1, 0, 1],
           [0, 0, 1, 1, 0]])

    A bow-tie (two triangles with only one node in common (no common edge).

    >>> bicycle(left_cycle=3, right_cycle=3, common_edges=0).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 0, 0],
           [1, 1, 0, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 0]])
    """
    assert (left_cycle - common_edges >= 2) and (right_cycle - common_edges >= 2)
    n = left_cycle + right_cycle - common_edges - 1
    adja = np.zeros([n, n], dtype=int)
    adja[:left_cycle, :left_cycle] = pan(cycle=left_cycle, tail=0).adjacency
    adja[(n - right_cycle):, (n - right_cycle):] = pan(cycle=right_cycle, tail=0).adjacency
    return SimpleGraph(adjacency=adja, names=names)


def dumbbells(left_cycle=3, center=1, right_cycle=3, names=None):
    """
    Parameters
    ----------
    left_cycle: :class:`int`
        Size of the first cycle.
    right_cycle: :class:`int`
        Size of the second cycle.
    center: :class:`int`
        Length of the path that joins the two cycles.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A graph with with two cycles joined by a path.

    Examples
    --------

    A square and a triangle joined by a path of length 3.

    >>> graph = dumbbells(left_cycle=4, right_cycle=3, center=3)
    >>> graph.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 0]])

    A bow-tie (two triangles with one node in common).

    >>> dumbbells(left_cycle=3, right_cycle=3, center=0).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 0, 0],
           [1, 1, 0, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 0]])
    """
    n = left_cycle + center + right_cycle - 1
    adja = np.zeros([n, n], dtype=int)
    adja[:(left_cycle + center), :(left_cycle + center)] = pan(cycle=left_cycle, tail=center).adjacency
    adja[(n - right_cycle):, (n - right_cycle):] = pan(cycle=right_cycle, tail=0).adjacency
    return SimpleGraph(adjacency=adja, names=names)


def triangle_chain(triangles=3, names=None):
    """
    Parameters
    ----------
    triangles: :class:`int`
        Number of triangles in the chain.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A graph made of a chain of triangles.

    Examples
    --------

    The Braess graph (two triangles).

    >>> braess = triangle_chain(triangles=2)
    >>> braess.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [0, 1, 1, 0]])

    The *Olympic Rings* graph.

    >>> triangle_chain(triangles=3).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 1, 0],
           [1, 1, 0, 1, 1],
           [0, 1, 1, 0, 1],
           [0, 0, 1, 1, 0]])
    """
    n = triangles + 2
    adja = np.zeros([n, n], dtype=int)
    adja[0, 1] = 1
    adja[1, 0] = 1
    for i in range(triangles):
        adja[i + 1, i + 2] = 1
        adja[i, i + 2] = 1
        adja[i + 2, i + 1] = 1
        adja[i + 2, i] = 1
    return SimpleGraph(adja, names=names)


def hyper_dumbbells(left_cycle=3, right_cycle=3, hyperedges=1, names=None):
    """
    Parameters
    ----------
    left_cycle: :class:`int`
        Size of the first cycle.
    right_cycle: :class:`int`
        Size of the second cycle.
    hyperedges: :class:`int`
        Length of the path of 3-edges that joins the two cycles.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.HyperGraph`
        Hypergraph of 2 regular cycles connected by a chain of 3-edges.

    Examples
    --------

    The *candy*, a basic but very useful hypergraph.

    >>> candy = hyper_dumbbells()
    >>> candy.incidence.toarray().astype('int') # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1, 0, 1],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 1, 0]])

    A more elaborate hypergraph

    >>> hyper_dumbbells(left_cycle=5, right_cycle=4, hyperedges=3).incidence.toarray().astype('int') # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]])

    Warning: without any hyperedge, we have two disconnected cycles.

    >>> hyper_dumbbells(hyperedges=0).incidence.toarray().astype('int') # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 1]])
    """
    n = left_cycle + right_cycle + hyperedges
    incidence = np.zeros((n, n), dtype=int)
    left = pan(cycle=left_cycle, tail=0).incidence.toarray()
    incidence[:left_cycle, :left_cycle] = left
    right = pan(cycle=right_cycle, tail=0).incidence.toarray()
    incidence[(n - right_cycle):, left_cycle:(left_cycle + right_cycle)] = right
    for i in range(hyperedges):
        incidence[(left_cycle - 1 + i):(left_cycle + 2 + i), n - hyperedges + i] = 1
    return HyperGraph(incidence=incidence, names=names)


def fan(cycles=3, cycle_size=3, hyperedges=1, names=None):
    """
    Parameters
    ----------
    cycles: :class:`int`
        Number of cycles
    cycle_size: :class:`int`
        Size of cycles
    hyperedges: :class:`int`
        Number of hyperedges that connect the cycles.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.HyperGraph`
        Return cycles connected by one hyperedge.

    Examples
    --------

    A basic 3-leaves clover:

    >>> clover = fan()
    >>> clover.incidence.toarray().astype('int')  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
           [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])

    Increase the hyperedge connectivity:

    >>> connected = fan(hyperedges=2)
    >>> connected.incidence.toarray().astype('int')  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]])

    With only two cycles, we have a simple graph.

    >>> db = fan(cycles=2, cycle_size=4)
    >>> db.incidence.toarray().astype('int') # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 0, 1],
           [0, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 0, 1, 0]])
    >>> db.to_simplegraph().adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 0, 1, 0]])
    """
    n = cycles * cycle_size
    incidence = np.zeros((n, n + hyperedges), dtype=int)
    for c in range(cycles):
        incidence[(c * cycle_size):((c + 1) * cycle_size),
        (c * cycle_size):((c + 1) * cycle_size)] = pan(cycle=cycle_size, tail=0).incidence.toarray()
        for h in range(hyperedges):
            incidence[c * cycle_size + h, h - hyperedges] = 1
    return HyperGraph(incidence=incidence, names=names)
