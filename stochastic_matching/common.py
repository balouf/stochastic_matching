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
    root: :class:`class`, optional
        Starting class (can be abstract).

    Returns
    -------
    :class:`dict`
        Dictionaries of all subclasses that have a name (as in class attribute `name`).

    Examples
    --------
    >>> from stochastic_matching.model import Model
    >>> get_classes(Model) # doctest: +NORMALIZE_WHITESPACE
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

    >>> neighbors(0, csr_matrix(incidence))
    array([0, 1, 2], dtype=int32)

    Egde of node 3:

    >>> neighbors(3, csr_matrix(incidence))
    array([2], dtype=int32)

    Nodes of edge 0:

    >>> neighbors(0, csc_matrix(incidence))
    array([0, 1], dtype=int32)

    Nodes of hyperedge 2:

    >>> neighbors(2, csc_matrix(incidence))
    array([0, 1, 2, 3], dtype=int32)
    """
    return compressed_incidence.indices[compressed_incidence.indptr[i]:compressed_incidence.indptr[i + 1]]
