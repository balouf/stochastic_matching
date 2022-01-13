import numpy as np
from stochastic_matching.graphs.classes import SimpleGraph, HyperGraph


def adja_maker_to_simple_graph(maker, *args, names=None, **kwargs):
    """
    Parameters
    ----------
    maker: callable
        Function that creates an adjacency matrix.
    args: :class:`list`
        Arguments of the maker.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)
    kwargs: :class:`dict`
        Keyword arguments of the maker.

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        Simple graph based on the adjacency matrix produced by the maker.
    """
    adja = maker(*args, **kwargs)
    return SimpleGraph(adjacency=adja, names=names)


def path_adjacency(n=2):
    """
    Parameters
    ----------
    n: :class:`int`
        Number of nodes.

    Returns
    -------
    :class:`~numpy.ndarray`
        Adjacency matrix of a path graph :math:`P_n` (cf https://mathworld.wolfram.com/PathGraph.html).
    """
    adja = np.zeros([n, n], dtype=int)
    for i in range(n-1):
        adja[i, i+1] = 1
        adja[i+1, i] = 1
    return adja


def path_graph(n=2, names=None):
    """
    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A path graph :math:`P_n` (cf https://mathworld.wolfram.com/PathGraph.html).

    Examples
    --------

    Default is a two nodes line:

    >>> p2 = path_graph()
    >>> p2.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1],
           [1, 0]])

    A five nodes line:

    >>> p5 = path_graph(5)
    >>> p5.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 0, 0],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [0, 0, 1, 0, 1],
           [0, 0, 0, 1, 0]])
    """
    return adja_maker_to_simple_graph(path_adjacency, names=names, n=n)


def cycle_adjacency(n=3):
    """
    Parameters
    ----------
    n: :class:`int`
        Length of the cycle.

    Returns
    -------
    :class:`~numpy.ndarray`
        Adjacency matrix of a cycle graph :math:`C_n` (cf https://mathworld.wolfram.com/CycleGraph.html).
    """
    adja = path_adjacency(n)
    adja[0, -1] = 1
    adja[-1, 0] = 1
    return adja


def cycle_graph(n=3, names=None):
    """
    Parameters
    ----------
    n: :class:`int`
        Length of the cycle.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display).

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A cycle graph :math:`C_n` (cf https://mathworld.wolfram.com/CycleGraph.html).

    Examples
    --------

    A triangle:

    >>> triangle = cycle_graph()
    >>> triangle.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]])

    A pentacle:

    >>> pentacle = cycle_graph(n=5)
    >>> pentacle.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 0, 1],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [0, 0, 1, 0, 1],
           [1, 0, 0, 1, 0]])

    A square:

    >>> square = cycle_graph(4)
    >>> square.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1],
           [1, 0, 1, 0],
           [0, 1, 0, 1],
           [1, 0, 1, 0]])
    """
    return adja_maker_to_simple_graph(cycle_adjacency, names=names, n=n)


def complete_adjacency(n=3):
    """
    Parameters
    ----------
    n: :class:`int`
        Number of nodes.

    Returns
    -------
    :class:`~numpy.ndarray`
        Adjacency matrix of a complete graph :math:`K_n` (cf https://mathworld.wolfram.com/CompleteGraph.html).
    """
    return np.ones([n, n], dtype=int) - np.identity(n, dtype=int)


def complete_graph(n=3, names=None):
    """
    Parameters
    ----------
    n: :class:`int`
        Length of the cycle.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display).

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A complete graph :math:`K_n` (cf https://mathworld.wolfram.com/CompleteGraph.html).

    Examples
    --------

    A triangle:

    >>> triangle = complete_graph()
    >>> triangle.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]])

    :math:`K_5`:

    >>> k5 = complete_graph(n=5)
    >>> k5.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 1, 1],
           [1, 0, 1, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 1, 0, 1],
           [1, 1, 1, 1, 0]])

    :math:`K_4`:

    >>> k4 = complete_graph(4)
    >>> k4.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [1, 1, 1, 0]])
    """
    return adja_maker_to_simple_graph(complete_adjacency, names=names, n=n)


def concatenate_adjacency(adja_list, overlap=None):
    """
    Parameters
    ----------
    adja_list: :class:`list` of :class:`~numpy.ndarray`
        The adjacency matrices that one wants to merge.
    overlap: :class:`int` or :class:`list` of :class:`int`, optional
        Number of nodes that are common to two consecutive graphs. Default to 1.

    Returns
    -------
    :class:`~numpy.ndarray`
        Concatenated adjacency.

    Examples
    --------

    A codomino adjacency:

    >>> concatenate_adjacency([cycle_adjacency(), cycle_adjacency(4), cycle_adjacency()], 2) # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 1, 0],
           [1, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 1],
           [0, 1, 0, 1, 0, 1],
           [0, 0, 0, 1, 1, 0]])
    """
    adja_list = [a for a in adja_list if a.shape[0] > 0]
    na = len(adja_list)
    if overlap is None:
        overlap = 1
    if type(overlap) == int:
        overlap = [overlap] * (na - 1)
    n = sum( adja.shape[0] for adja in adja_list) - sum(overlap)
    adja = np.zeros([n, n], dtype=int)
    c_a = adja_list[0]
    c_n = c_a.shape[0]
    adja[:c_n, :c_n] = c_a
    pointer = c_n
    for i, o in enumerate(overlap):
        pointer -= o
        c_a = adja_list[i+1]
        c_n = c_a.shape[0]
        adja[pointer:pointer+c_n, pointer:pointer+c_n] = c_a
        pointer += c_n
    return adja


def concatenate(graph_list, overlap=None, names=None):
    """
    Parameters
    ----------
    graph_list: :class:`list` of :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        The graphs that one want to merge.
    overlap: :class:`int` or :class:`list` of :class:`int`, optional
        Number of nodes that are common to two consecutive graphs. Default to 1.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display).

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        Concatenated graph.

    Examples
    --------

    A codomino graph:

    >>> codomino = concatenate([cycle_graph(), cycle_graph(4), cycle_graph()], 2)
    >>> codomino.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 1, 0],
           [1, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 1],
           [0, 1, 0, 1, 0, 1],
           [0, 0, 0, 1, 1, 0]])
    """
    adja_list = [g.adjacency for g in graph_list]
    return adja_maker_to_simple_graph(concatenate_adjacency, names=names, adja_list=adja_list, overlap=overlap)


def tadpole_graph(m=3, n=1, names=None):
    """
    Parameters
    ----------
    m: :class:`int`
        Length of the cycle.
    n: :class:`int`
        Length of the tail.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display).

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A graph with tail of length *n* attached to a cycle of length *m*, aka a tadpole graph :math:`T_(m, n)`
        (cf https://mathworld.wolfram.com/TadpoleGraph.html).

    Examples
    --------

    A triangle with a one-edge tail (paw graph):

    >>> paw = tadpole_graph()
    >>> paw.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 1],
           [0, 0, 1, 0]])

    A pentacle:

    >>> c5 = tadpole_graph(m=5, n=0)
    >>> c5.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 0, 1],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [0, 0, 1, 0, 1],
           [1, 0, 0, 1, 0]])

    A larger tadpole:

    >>> long_pan = tadpole_graph(m=4, n=3)
    >>> long_pan.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 0, 1, 0]])
    """
    adja = concatenate_adjacency(adja_list=[cycle_adjacency(m), path_adjacency(n + 1)], overlap=1)
    return SimpleGraph(adjacency=adja, names=names)


def lollipop_graph(m=3, n=1, names=None):
    """
    Parameters
    ----------
    m: :class:`int`
        Length of the complete graph.
    n: :class:`int`
        Length of the tail.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display).

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A tail of length *n* attached to a complete graph of size *m*, aka a lollipop graph
        (cf https://mathworld.wolfram.com/LollipopGraph.html).

    Examples
    --------

    A triangle with a one-edge tail (paw graph):

    >>> l_3_1 = lollipop_graph()
    >>> l_3_1.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 1],
           [0, 0, 1, 0]])


    A larger lollipop:

    >>> l_5_3 = lollipop_graph(m=5, n=3)
    >>> l_5_3.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 1, 1, 0, 0, 0],
           [1, 0, 1, 1, 1, 0, 0, 0],
           [1, 1, 0, 1, 1, 0, 0, 0],
           [1, 1, 1, 0, 1, 0, 0, 0],
           [1, 1, 1, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 0, 0, 1, 0]])
    """
    adja = concatenate_adjacency(adja_list=[complete_adjacency(m), path_adjacency(n + 1)], overlap=1)
    return SimpleGraph(adjacency=adja, names=names)


def bicycle_graph(left_cycle=3, right_cycle=3, common_edges=1, names=None):
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

    >>> house = bicycle_graph(left_cycle=4, right_cycle=3, common_edges=1)
    >>> house.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 0],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 1],
           [1, 0, 1, 0, 1],
           [0, 0, 1, 1, 0]])

    A bow-tie (two triangles with only one node in common (no common edge).

    >>> bicycle_graph(left_cycle=3, right_cycle=3, common_edges=0).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 0, 0],
           [1, 1, 0, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 0]])
    """
    assert (left_cycle - common_edges >= 2) and (right_cycle - common_edges >= 2)
    adja = concatenate_adjacency([cycle_adjacency(left_cycle), cycle_adjacency(right_cycle)], overlap=(common_edges+1))
    return SimpleGraph(adjacency=adja, names=names)


def kayak_paddle_graph(k=3, l=1, m=3, names=None):
    """
    Parameters
    ----------
    k: :class:`int`
        Size of the first cycle.
    m: :class:`int`
        Size of the second cycle.
    l: :class:`int`
        Length of the path that joins the two cycles.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A graph with with two cycles joined by a path, aka a kayak paddle graph :math:`KP(k, m, l)`
        (cf https://mathworld.wolfram.com/KayakPaddleGraph.html).

    Examples
    --------

    A square and a triangle joined by a path of length 3.

    >>> graph = kayak_paddle_graph(k=4, m=3, l=3)
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

    >>> kayak_paddle_graph(k=3, m=3, l=0).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 0, 0],
           [1, 1, 0, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 0]])
    """
    adja = concatenate_adjacency([cycle_adjacency(k), path_adjacency(l + 1), cycle_adjacency(m)])
    return SimpleGraph(adjacency=adja, names=names)



def barbell_graph(k=3, l=1, m=None, names=None):
    """
    Parameters
    ----------
    k: :class:`int`
        Size of the first complete graph.
    m: :class:`int`, optional
        Size of the second complete graph. Default to the size of the first complete graph.
    l: :class:`int`
        Length of the path that joins the two complete graphs.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        A barbell graph, aka two complete graphs bridged by a path
        (cf https://mathworld.wolfram.com/BarbellGraph.html).

    Examples
    --------

    Traditional barbel graph with complete graphs of same size and unit bridge.

    >>> barbel_5 = barbell_graph(5)
    >>> barbel_5.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]])

    A bow-tie (two triangles with one node in common).

    >>> barbell_graph(l=0).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 0, 0],
           [1, 1, 0, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 0]])

    Something more elaborated.

    >>> barbell_graph(k=3, m=5, l=4).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]])
    """
    if m is None:
        m = k
    adja = concatenate_adjacency([complete_adjacency(k), path_adjacency(l + 1), complete_adjacency(m)])
    return SimpleGraph(adjacency=adja, names=names)


def chained_cycle_graph(n=3, c=3, o=2, names=None):
    """
    Parameters
    ----------
    n: :class:`int`
        Size of the cycles.
    c: :class:`int`
        Number of copies.
    o: :class:`int`
        Overlap between cycles.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)

    Returns
    -------
    :class:`~stochastic_matching.graphs.classes.SimpleGraph`
        Concatenate *c* copies of :math:`C_n`, with *o* overlapping nodes between two consecutive cycles.

    Examples
    --------

    The diamond graph (two triangles).

    >>> diamond = chained_cycle_graph(n=3, c=2)
    >>> diamond.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [0, 1, 1, 0]])

    The *triamond* graph.

    >>> chained_cycle_graph(n=3, c=3).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 1, 0],
           [1, 1, 0, 1, 1],
           [0, 1, 1, 0, 1],
           [0, 0, 1, 1, 0]])

    The triangular snake graph :math:`TS_9` (cf https://mathworld.wolfram.com/TriangularSnakeGraph.html)

    >>> chained_cycle_graph(n=3, c=4, o=1).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 0]])

    The domino graph, or 3-ladder graph (cf https://mathworld.wolfram.com/LadderGraph.html)

    >>> chained_cycle_graph(n=4, c=2).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 0, 0],
           [1, 0, 1, 0, 0, 0],
           [0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0],
           [0, 0, 0, 1, 0, 1],
           [0, 0, 1, 0, 1, 0]])
    """
    adja = concatenate_adjacency([cycle_adjacency(n)]*c, o )
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

    The diamond graph (two triangles).

    >>> diamond = triangle_chain(triangles=2)
    >>> diamond.adjacency # doctest: +NORMALIZE_WHITESPACE
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


def hyper_paddle(left_cycle=3, right_cycle=3, hyperedges=1, names=None):
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

    >>> candy = hyper_paddle()
    >>> candy.incidence.toarray().astype('int') # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1, 0, 1],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 1, 0]])

    A more elaborate hypergraph

    >>> hyper_paddle(left_cycle=5, right_cycle=4, hyperedges=3).incidence.toarray().astype('int') # doctest: +NORMALIZE_WHITESPACE
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

    >>> hyper_paddle(hyperedges=0).incidence.toarray().astype('int') # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 1]])
    """
    n = left_cycle + right_cycle + hyperedges
    incidence = np.zeros((n, n), dtype=int)
    left = cycle_graph(n=left_cycle).incidence.toarray()
    incidence[:left_cycle, :left_cycle] = left
    right = cycle_graph(n=right_cycle).incidence.toarray()
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
    cycle_incidence = cycle_graph(n=cycle_size).incidence.toarray()
    for c in range(cycles):
        incidence[(c * cycle_size):((c + 1) * cycle_size),
        (c * cycle_size):((c + 1) * cycle_size)] = cycle_incidence
        for h in range(hyperedges):
            incidence[c * cycle_size + h, h - hyperedges] = 1
    return HyperGraph(incidence=incidence, names=names)
