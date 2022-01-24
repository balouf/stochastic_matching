import numpy as np

from stochastic_matching.model import Model


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
    for i in range(n - 1):
        adja[i, i + 1] = 1
        adja[i + 1, i] = 1
    return adja


class Path(Model):
    """
    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------

    Default is a two nodes line:

    >>> p2 = Path()
    >>> p2.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1],
           [1, 0]])

    Class name:

    >>> p2.name
    'Path'

    A five nodes line with alphabetical names:

    >>> p5 = Path(5, names="alpha")
    >>> p5.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 0, 0],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [0, 0, 1, 0, 1],
           [0, 0, 0, 1, 0]])

    >>> [p5.names[i] for i in range(p5.n)]
    ['A', 'B', 'C', 'D', 'E']
    """
    name = "Path"

    def __init__(self, n=2, *args, **kwargs):
        adja = path_adjacency(n)
        super(Path, self).__init__(adjacency=adja, *args, **kwargs)


def star_adjacency(n=4):
    """
    Parameters
    ----------
    n: :class:`int`
        Number of nodes.

    Returns
    -------
    :class:`~numpy.ndarray`
        Adjacency matrix of a star graph :math:`S_n` (https://mathworld.wolfram.com/StarGraph.html).
    """
    adja = np.zeros([n, n], dtype=int)
    for i in range(n - 1):
        adja[0, i + 1] = 1
        adja[i + 1, 0] = 1
    return adja


class Star(Model):
    """
    Parameters
    ----------

    n: :class:`int`
        Number of nodes.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------

    Default is a 4 nodes star (one center and three branches):

    >>> s4 = Star()
    >>> s4.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 1],
           [1, 0, 0, 0],
           [1, 0, 0, 0],
           [1, 0, 0, 0]])

    A five nodes star:

    >>> s5 = Star(5)
    >>> s5.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 1, 1],
           [1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0]])
    """
    name = "Star"

    def __init__(self, n=4, *args, **kwargs):
        adja = star_adjacency(n)
        super(Star, self).__init__(adjacency=adja, *args, **kwargs)


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


class Cycle(Model):
    """
    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------

    A triangle:

    >>> triangle = Cycle()
    >>> triangle.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]])

    A pentacle:

    >>> pentacle = Cycle(n=5)
    >>> pentacle.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 0, 1],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [0, 0, 1, 0, 1],
           [1, 0, 0, 1, 0]])

    A square:

    >>> square = Cycle(4)
    >>> square.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1],
           [1, 0, 1, 0],
           [0, 1, 0, 1],
           [1, 0, 1, 0]])
    """
    name = "Cycle"

    def __init__(self, n=3, *args, **kwargs):
        adja = cycle_adjacency(n)
        super(Cycle, self).__init__(adjacency=adja, *args, **kwargs)


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


class Complete(Model):
    """
    Parameters
    ----------
    n: :class:`int`
        Length of the cycle.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------

    A triangle:

    >>> triangle = Complete()
    >>> triangle.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]])

    :math:`K_5`:

    >>> k5 = Complete(n=5)
    >>> k5.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 1, 1],
           [1, 0, 1, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 1, 0, 1],
           [1, 1, 1, 1, 0]])

    :math:`K_4`:

    >>> k4 = Complete(4)
    >>> k4.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [1, 1, 1, 0]])
    """
    name = "Complete"

    def __init__(self, n=3, *args, **kwargs):
        adja = complete_adjacency(n)
        super(Complete, self).__init__(adjacency=adja, *args, **kwargs)


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
    n = sum(adja.shape[0] for adja in adja_list) - sum(overlap)
    adja = np.zeros([n, n], dtype=int)
    c_a = adja_list[0]
    c_n = c_a.shape[0]
    adja[:c_n, :c_n] = c_a
    pointer = c_n
    for i, o in enumerate(overlap):
        pointer -= o
        c_a = adja_list[i + 1]
        c_n = c_a.shape[0]
        adja[pointer:pointer + c_n, pointer:pointer + c_n] = c_a
        pointer += c_n
    return adja


def concatenate(model_list, overlap=None, *args, **kwargs):
    """
    Parameters
    ----------
    model_list: :class:`list` of :class:`~stochastic_matching.model.Model`
        The graphs that one want to merge. They must all be simple (and in particular have an adjacency matrix).
    overlap: :class:`int` or :class:`list` of :class:`int`, optional
        Number of nodes that are common to two consecutive graphs. Default to 1.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Returns
    -------
    :class:`~stochastic_matching.model.Model`
        Model of the concatenated graph.

    Examples
    --------

    A codomino graph:

    >>> codomino = concatenate([Cycle(), Cycle(4), Cycle()], 2)
    >>> codomino.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 1, 0],
           [1, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 1],
           [0, 1, 0, 1, 0, 1],
           [0, 0, 0, 1, 1, 0]])
    """
    adja_list = [g.adjacency for g in model_list]
    adja = concatenate_adjacency(adja_list, overlap)
    return Model(adjacency=adja, *args, **kwargs)


class Codomino(Model):
    """
    Parameters
    ----------
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------
    >>> codomino = Codomino()
    >>> codomino.adjacency
    array([[0, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 1, 0],
           [1, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 1],
           [0, 1, 0, 1, 0, 1],
           [0, 0, 0, 1, 1, 0]])
    """
    name = "Codomino"

    def __init__(self, *args, **kwargs):
        adja = concatenate_adjacency([cycle_adjacency(), cycle_adjacency(4), cycle_adjacency()], 2)
        super(Codomino, self).__init__(adjacency=adja, *args, **kwargs)


class Pyramid(Model):
    """
    Parameters
    ----------
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------
    >>> pyramid = Pyramid()
    >>> pyramid.adjacency
    array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
           [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
           [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])
    """
    name = "Pyramid"

    def __init__(self, *args, **kwargs):
        adja = concatenate_adjacency([cycle_adjacency(), cycle_adjacency(5),
                                      cycle_adjacency(5), cycle_adjacency()], 2)
        super(Pyramid, self).__init__(adjacency=adja, *args, **kwargs)


class Tadpole(Model):
    """
    Parameters
    ----------
    m: :class:`int`
        Length of the cycle.
    n: :class:`int`
        Length of the tail.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.


    Examples
    --------

    A triangle with a one-edge tail (paw graph):

    >>> paw = Tadpole()
    >>> paw.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 1],
           [0, 0, 1, 0]])

    A pentacle:

    >>> c5 = Tadpole(m=5, n=0)
    >>> c5.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 0, 1],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [0, 0, 1, 0, 1],
           [1, 0, 0, 1, 0]])

    A larger tadpole:

    >>> long_pan = Tadpole(m=4, n=3)
    >>> long_pan.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 0, 1, 0]])
    """
    name = "Tadpole"

    def __init__(self, m=3, n=1, *args, **kwargs):
        adja = concatenate_adjacency(adja_list=[cycle_adjacency(m), path_adjacency(n + 1)], overlap=1)
        super(Tadpole, self).__init__(adjacency=adja, *args, **kwargs)


class Lollipop(Model):
    """
    Parameters
    ----------
    m: :class:`int`
        Length of the complete graph.
    n: :class:`int`
        Length of the tail.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------

    A triangle with a one-edge tail (paw graph):

    >>> l_3_1 = Lollipop()
    >>> l_3_1.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 1],
           [0, 0, 1, 0]])


    A larger lollipop:

    >>> l_5_3 = Lollipop(m=5, n=3)
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
    name = "Lollipop"

    def __init__(self, m=3, n=1, *args, **kwargs):
        adja = concatenate_adjacency(adja_list=[complete_adjacency(m), path_adjacency(n + 1)], overlap=1)
        super(Lollipop, self).__init__(adjacency=adja, *args, **kwargs)


class KayakPaddle(Model):
    """
    Parameters
    ----------
    k: :class:`int`
        Size of the first cycle.
    l: :class:`int`
        Length of the path that joins the two cycles.
    m: :class:`int`, optional
        Size of the second cycle. Default to the size of the first cycle.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------

    A square and a triangle joined by a path of length 3.

    >>> kayak = KayakPaddle(k=4, m=3, l=3)
    >>> kayak.adjacency # doctest: +NORMALIZE_WHITESPACE
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

    >>> KayakPaddle(l=0).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 0, 0],
           [1, 1, 0, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 0]])
    """
    name = "Kayak Paddle"

    def __init__(self, k=3, l=1, m=None, *args, **kwargs):
        if m is None:
            m = k
        adja = concatenate_adjacency([cycle_adjacency(k), path_adjacency(l + 1), cycle_adjacency(m)])
        super(KayakPaddle, self).__init__(adjacency=adja, *args, **kwargs)


class Barbell(Model):
    """
    Parameters
    ----------
    k: :class:`int`
        Size of the first complete graph.
    m: :class:`int`, optional
        Size of the second complete graph. Default to the size of the first complete graph.
    l: :class:`int`
        Length of the path that joins the two complete graphs.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------

    Traditional barbel graph with complete graphs of same size and unit bridge.

    >>> barbel_5 = Barbell(5)
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

    >>> Barbell(l=0).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 0, 0],
           [1, 1, 0, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 0]])

    Something more elaborated.

    >>> Barbell(k=3, m=5, l=4).adjacency # doctest: +NORMALIZE_WHITESPACE
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
    name = "Barbell"

    def __init__(self, k=3, l=1, m=None, *args, **kwargs):
        if m is None:
            m = k
        adja = concatenate_adjacency([complete_adjacency(k), path_adjacency(l + 1), complete_adjacency(m)])
        super(Barbell, self).__init__(adjacency=adja, *args, **kwargs)


class CycleChain(Model):
    """
    Parameters
    ----------
    n: :class:`int`
        Size of the cycles.
    c: :class:`int`
        Number of copies.
    o: :class:`int`
        Overlap between cycles.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------

    The diamond graph (two triangles).

    >>> diamond = CycleChain()
    >>> diamond.adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [0, 1, 1, 0]])

    The *triamond* graph.

    >>> CycleChain(n=3, c=3).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 1, 0, 0],
           [1, 0, 1, 1, 0],
           [1, 1, 0, 1, 1],
           [0, 1, 1, 0, 1],
           [0, 0, 1, 1, 0]])

    The triangular snake graph :math:`TS_9` (cf https://mathworld.wolfram.com/TriangularSnakeGraph.html)

    >>> CycleChain(n=3, c=4, o=1).adjacency # doctest: +NORMALIZE_WHITESPACE
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

    >>> CycleChain(n=4, c=2).adjacency # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 0, 0],
           [1, 0, 1, 0, 0, 0],
           [0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0],
           [0, 0, 0, 1, 0, 1],
           [0, 0, 1, 0, 1, 0]])
    """
    name = "Cycle Chain"

    def __init__(self, n=3, c=2, o=2, *args, **kwargs):
        adja = concatenate_adjacency([cycle_adjacency(n)] * c, o)
        super(CycleChain, self).__init__(adjacency=adja, *args, **kwargs)


class HyperPaddle(Model):
    """
    Parameters
    ----------
    k: :class:`int`
        Size of the first cycle.
    m: :class:`int`
        Size of the second cycle.
    l: :class:`int`
        Length of the path of 3-edges that joins the two cycles.
    *args
        Positional parameters for the model.
    **kwargs: :class:`dict`, optional
        Keyword parameters for the model.

    Examples
    --------

    The *candy*, a basic but very useful hypergraph.

    >>> candy = HyperPaddle()
    >>> candy.incidence
    array([[1, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1, 0, 1],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 1, 0]])

    A more elaborate hypergraph

    >>> HyperPaddle(k=5, m=4, l=3).incidence
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

    >>> HyperPaddle(l=0).incidence
    array([[1, 1, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 1]])
    """
    name = "Hyper Kayak Paddle"

    def __init__(self, k=3, m=3, l=1, *args, **kwargs):
        n = k + m + l
        incidence = np.zeros((n, n), dtype=int)
        left = Cycle(n=k).incidence
        incidence[:k, :k] = left
        right = Cycle(n=m).incidence
        incidence[(n - m):, k:(k + m)] = right
        for i in range(l):
            incidence[(k - 1 + i):(k + 2 + i), n - l + i] = 1
        super(HyperPaddle, self).__init__(incidence=incidence, *args, **kwargs)


class Fan(Model):
    """
    Parameters
    ----------
    cycles: :class:`int`
        Number of cycles
    cycle_size: :class:`int`
        Size of cycles
    hyperedges: :class:`int`
        Number of hyperedges that connect the cycles.
    *args
        Positional parameters for the model.
    **kwargs
        Keyword parameters for the model.

    Examples
    --------

    A basic 3-leaves clover:

    >>> clover = Fan()
    >>> clover.incidence  # doctest: +NORMALIZE_WHITESPACE
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

    >>> connected = Fan(hyperedges=2)
    >>> connected.incidence  # doctest: +NORMALIZE_WHITESPACE
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

    >>> db = Fan(cycles=2, cycle_size=4)
    >>> db.incidence # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 0, 1],
           [0, 0, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 0, 1, 0]])
    >>> from stochastic_matching.model import incidence_to_adjacency
    >>> incidence_to_adjacency(db.incidence) # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 0, 1, 0]])
    """
    name = "Fan"

    def __init__(self, cycles=3, cycle_size=3, hyperedges=1, *args, **kwargs):
        n = cycles * cycle_size
        incidence = np.zeros((n, n + hyperedges), dtype=int)
        cycle_incidence = Cycle(n=cycle_size).incidence
        for c in range(cycles):
            incidence[(c * cycle_size):((c + 1) * cycle_size),
            (c * cycle_size):((c + 1) * cycle_size)] = cycle_incidence
            for h in range(hyperedges):
                incidence[c * cycle_size + h, h - hyperedges] = 1
        super(Fan, self).__init__(incidence=incidence, *args, **kwargs)

