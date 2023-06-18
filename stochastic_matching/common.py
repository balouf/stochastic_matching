from inspect import isclass

import numpy as np
from numba import njit

import os
if os.environ.get("NUMBA_DISABLE_JIT") == "1":
    List = lambda x: x
else:
    from numba.typed import List


def pseudo_inverse_scalar(x):
    """
    Parameters
    ----------
    x: :class:`float`

    Returns
    -------
    :class:`float`
        Inverse of x if it is not 0.

    Examples
    --------

    >>> pseudo_inverse_scalar(2.0)
    0.5
    >>> pseudo_inverse_scalar(0)
    0.0
    """
    return 0.0 if x == 0 else 1 / x


def clean_zeros(matrix, tol=1e-10):
    """
    Replace in-place all small values of a matrix by 0.

    Parameters
    ----------
    matrix: :class:`~numpy.ndarray`
        Matrix to clean.
    tol: :class:`float`, optional
        Threshold. All entries with absolute value lower than `tol` are put to zero.

    Returns
    -------
    None

    Examples
    --------

    >>> import numpy as np
    >>> mat = np.array([[1e-12, -.3], [.8, -1e-13]])
    >>> clean_zeros(mat)
    >>> mat # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. , -0.3],
           [ 0.8,  0. ]])
    """
    matrix[abs(matrix[:]) < tol] = 0


class CharMaker:
    """
    Class that acts as an infinite list of letters. Used to provide letter-labels to nodes

    Examples
    --------

    >>> names = CharMaker()
    >>> names[0]
    'A'
    >>> names[7]
    'H'
    >>> names[26]
    'AA'
    >>> names[107458610947716]
    'STOCHASTIC'
    """

    def __init__(self):
        pass

    @staticmethod
    def to_char(i):
        return chr(ord('A') + (i % 26))

    def __getitem__(self, i):
        res = self.to_char(i)
        while i > 25:
            i = i // 26 - 1
            res = f"{self.to_char(i)}{res}"
        return res


def get_classes(root):
    """
    Parameters
    ----------
    root: :class:`class`
        Starting class (can be abstract).

    Returns
    -------
    :class:`dict`
        Dictionaries of all subclasses that have a name (as in class attribute `name`).

    Examples
    --------
    >>> import stochastic_matching as sm
    >>> get_classes(sm.Model) # doctest: +NORMALIZE_WHITESPACE
    {'Path': <class 'stochastic_matching.graphs.Path'>,
     'Star': <class 'stochastic_matching.graphs.Star'>,
     'Cycle': <class 'stochastic_matching.graphs.Cycle'>,
     'Complete': <class 'stochastic_matching.graphs.Complete'>,
     'Codomino': <class 'stochastic_matching.graphs.Codomino'>,
     'Pyramid': <class 'stochastic_matching.graphs.Pyramid'>,
     'Tadpole': <class 'stochastic_matching.graphs.Tadpole'>,
     'Lollipop': <class 'stochastic_matching.graphs.Lollipop'>,
     'Kayak Paddle': <class 'stochastic_matching.graphs.KayakPaddle'>,
     'Barbell': <class 'stochastic_matching.graphs.Barbell'>,
     'Cycle Chain': <class 'stochastic_matching.graphs.CycleChain'>,
     'Hyper Kayak Paddle': <class 'stochastic_matching.graphs.HyperPaddle'>,
     'Fan': <class 'stochastic_matching.graphs.Fan'>}
    """
    result = {c.name: c for c in root.__subclasses__() if c.name}
    for c in root.__subclasses__():
        result.update(get_classes(c))
    return result


def class_converter(subclass, motherclass):
    """
    Parameters
    ----------
    subclass: :class:`str` or :class:`class`
        Required subclass, or its name.
    motherclass: :class:`class`
        Ancestor of the subclass.

    Returns
    -------
    :class:`class`
        The subclass.

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> from stochastic_matching.simulator.generic import Simulator
    >>> class_converter('random_node', Simulator)
    <class 'stochastic_matching.simulator.size_based.RandomNodeSimulator'>
    >>> class_converter(sm.FCFM, Simulator)
    <class 'stochastic_matching.simulator.age_based.FCFM'>
    >>> class_converter('anything', Simulator)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ValueError: anything is not a known name for a subclass of Simulator.
    Known names: virtual_queue, generic_queue_size, fcfm, random_node, longest_queue, random_item, filtering, priority.

    >>> class_converter(2, Simulator)
    Traceback (most recent call last):
    ...
    TypeError: Subclass must be string or Simulator subclass (not instance).
    """
    if isinstance(subclass, str):
        class_dict = get_classes(motherclass)
        if subclass in class_dict:
            return class_dict[subclass]
        else:
            raise ValueError(f"{subclass} is not a known name for a subclass of {motherclass.__name__}. "
                             f"Known names: {', '.join(class_dict.keys())}.")
    elif isclass(subclass) and issubclass(subclass, motherclass):
        return subclass
    else:
        raise TypeError(f"Subclass must be string or {motherclass.__name__} subclass (not instance).")


def neighbors(i, compressed_incidence):
    """
    Return neighbors of a node/edge with respect to an incident matrix.
    Neighborhood is defined on hypergraph level, not on adjacency level:
    neighbors of a node are edges, neighbors of an edge are nodes.

    Parameters
    ----------
    i: :class:`int`
        Index of the node/edge to probe.
    compressed_incidence: :class:`~scipy.sparse.csr_matrix` or :class:`~scipy.sparse.csc_matrix`
        Compressed sparse incidence matrix on rows (for nodes) or columns (for edges).

    Returns
    -------
    :class:`~numpy.ndarray`
        Neighbors of *i*.

    Examples
    --------

    A hypergraph with 4 nodes, 2 regular edges (0, 1) and (0, 2) and one 4-edge (0, 1, 2, 3).

    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix, csc_matrix
    >>> incidence = np.array([[1, 1, 1],
    ...                       [1, 0, 1],
    ...                       [0, 1, 1],
    ...                       [0, 0, 1]])

    Edges of node 0:

    >>> neighbors(0, csr_matrix(incidence)) # doctest: +ELLIPSIS
    array([0, 1, 2]...)

    Egde of node 3:

    >>> neighbors(3, csr_matrix(incidence)) # doctest: +ELLIPSIS
    array([2]...)

    Nodes of edge 0:

    >>> neighbors(0, csc_matrix(incidence)) # doctest: +ELLIPSIS
    array([0, 1]...)

    Nodes of hyperedge 2:

    >>> neighbors(2, csc_matrix(incidence)) # doctest: +ELLIPSIS
    array([0, 1, 2, 3]...)
    """
    return compressed_incidence.indices[compressed_incidence.indptr[i]:compressed_incidence.indptr[i + 1]]


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

    Examples
    --------

    >>> set_seed(42)
    >>> np.random.rand()
    0.3745401188473625
    """
    np.random.seed(value)


def create_prob_alias(mu):
    """
    Prepare vector to draw a distribution with the alias method.

    Based on https://www.keithschwarz.com/darts-dice-coins/.

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


def graph_neighbors_list(model, forbidden_edges=None):
    """
    Extract Numba-compatible neighboring structures from a :class:`~stochastic_matching.model.Model`.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model with the graph to transform.


    Returns
    -------
    :class:`~numba.typed.List`
        List of neighbors for each node. For one given node yields a list of tuples where the first element is the edge
        and the second element the neighbor (for `SimpleGraph`) / array of neighbors (for `HyperGraph`).

    Examples
    --------

    Consider a diamond graph.

    >>> import stochastic_matching as sm
    >>> diamond = sm.CycleChain()
    >>> diamond.incidence
    array([[1, 1, 0, 0, 0],
           [1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1],
           [0, 0, 0, 1, 1]])

    Node 0 is connected with edge 0 to node 1 and with edge 1 to node 2.

    Node 1 is connected with edge 0 to node 0, with edge 2 to node 2, and with edge 3 to node 3.

    Node 2 is connected with edge 1 to node 0, with edge 2 to node 1, and with edge 4 to node 3.

    Node 3 is connected with edge 3 to node 1 and with edge 4 to node 2.

    This is exactly what `graph_neighbors_list` outputs, in a numba-compatible way.

    >>> graph_neighbors_list(diamond) # doctest: +NORMALIZE_WHITESPACE
    [[(0, 1), (1, 2)],
    [(0, 0), (2, 2), (3, 3)],
    [(1, 0), (2, 1), (4, 3)],
    [(3, 1), (4, 2)]]

    To remove some egdes, use the optional parameter forbidden_edges.

    >>> graph_neighbors_list(diamond, forbidden_edges={0, 4}) # doctest: +NORMALIZE_WHITESPACE
    [[(1, 2)],
     [(2, 2), (3, 3)],
     [(1, 0), (2, 1)],
     [(3, 1)]]

    With hypergraph notation, the second term of the tuples is an array instead of an integer.

    >>> diamond.adjacency = None
    >>> g = graph_neighbors_list(diamond)
    >>> [ [(e, a.astype(int)) for e, a in n] for n in g ] # doctest: +NORMALIZE_WHITESPACE
    [[(0, array([1])), (1, array([2]))],
    [(0, array([0])), (2, array([2])), (3, array([3]))],
    [(1, array([0])), (2, array([1])), (4, array([3]))],
    [(3, array([1])), (4, array([2]))]]


    Having arrays is only useful for true hypergraphs. For instance, in the candy hypergraph, edge 6 links
    nodes 2, 3, and 4 together.

    >>> candy = sm.HyperPaddle()
    >>> candy.incidence
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

    >>> g = graph_neighbors_list(candy, forbidden_edges={1, 3})
    >>> [ [(e, a.astype(int)) for e, a in n] for n in g ] # doctest: +NORMALIZE_WHITESPACE
    [[(0, array([1]))],
     [(0, array([0])), (2, array([2]))],
     [(2, array([1])), (6, array([3, 4]))],
     [(6, array([2, 4]))],
     [(4, array([6])), (6, array([2, 3]))],
     [(5, array([6]))],
     [(4, array([4])), (5, array([5]))]]

    """
    if forbidden_edges is None:
        forbidden_edges = set()
    edges = [model.incidence_csr.indices[model.incidence_csr.indptr[i]:model.incidence_csr.indptr[i + 1]]
             for i in range(model.n)]
    edges = [ [e for e in edges[i] if e not in forbidden_edges] for i in range(model.n) ]
    if model.adjacency is not None:
        neighbors = [[[k for k in model.incidence_csc.indices[model.incidence_csc.indptr[e]:model.incidence_csc.indptr[e + 1]] if k != i][0]
                      for e in edges[i]] for i in range(model.n)]
    else:
        neighbors = [
            [np.array([k for k in model.incidence_csc.indices[model.incidence_csc.indptr[e]:model.incidence_csc.indptr[e + 1]] if k != i], dtype=np.int32)
             for e in edges[i] if e not in forbidden_edges] for i in range(model.n)]
    return List([List([(e, v) for e, v in zip(edges[i], neighbors[i])]) for i in range(model.n)])
