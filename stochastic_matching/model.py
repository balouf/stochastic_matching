import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from cached_property import cached_property

from stochastic_matching.common import pseudo_inverse_scalar, clean_zeros, CharMaker, neighbors, class_converter
from stochastic_matching.display import show
from stochastic_matching.simulator.generic import Simulator

status_names = {(False, False): "Nonjective",
                (True, False): "Injective-only",
                (False, True): "Surjective-only",
                (True, True): "Bijective"}
"""Name associated to a (injective, surjective) tuple."""

status_names_simple_connected = {(False, False): "Bipartite with cycle(s)",
                                 (True, False): "Tree",
                                 (False, True): "Non-bipartite polycyclic",
                                 (True, True): "Non-bipartite monocyclic"}
"""Name associated to a (injective, surjective) tuple when the graph is simple and connected."""


def adjacency_to_incidence(adjacency):
    """
    Converts adjacency matrix to incidence matrix.

    Parameters
    ----------
    adjacency: :class:`~numpy.ndarray`
        Adjacency matrix of a simple graph (symmetric matrix with 0s and 1s, null diagonal).

    Returns
    -------
    :class:`~numpy.ndarray`
        Incidence matrix between nodes and edges.

    Examples
    --------

    Convert a diamond adjacency to incidence.

    >>> diamond = np.array([[0, 1, 1, 0],
    ...           [1, 0, 1, 1],
    ...           [1, 1, 0, 1],
    ...           [0, 1, 1, 0]])
    >>> adjacency_to_incidence(diamond)
    array([[1, 1, 0, 0, 0],
           [1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1],
           [0, 0, 0, 1, 1]])
    """
    n, _ = adjacency.shape
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j]]
    m = len(edges)
    incidence = np.zeros((n, m), dtype=int)
    for j, e in enumerate(edges):
        for i in e:
            incidence[i, j] = 1
    return incidence


def incidence_to_adjacency(incidence):
    """
    Converts incidence matrix to adjacency matrix.
    If the incidence matrix does not corresponds to a simple graph, an error is thrown.

    Parameters
    ----------
    incidence: :class:`~numpy.ndarray`
        Incidence matrix of a simple graph (matrix with 0s and 1s, two 1s per column).

    Returns
    -------
    :class:`~numpy.ndarray`
        Adjacency matrix.

    Examples
    --------

    Convert a diamond graph from incidence to adjacency.

    >>> import stochastic_matching as sm
    >>> diamond = sm.CycleChain()
    >>> diamond.incidence
    array([[1, 1, 0, 0, 0],
           [1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1],
           [0, 0, 0, 1, 1]])
    >>> incidence_to_adjacency(diamond.incidence)
    array([[0, 1, 1, 0],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [0, 1, 1, 0]])

    An error occurs if one tries to convert a hypergraph.

    >>> candy = sm.HyperPaddle()
    >>> candy.incidence
    array([[1, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1, 0, 1],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 1, 0]])
    >>> incidence_to_adjacency(candy.incidence)
    Traceback (most recent call last):
    ...
    ValueError: The incidence matrix does not seem to correspond to a simple graph.
    """
    # noinspection PyUnresolvedReferences
    if not np.all(np.sum(incidence, axis=0) == 2):
        raise ValueError("The incidence matrix does not seem to correspond to a simple graph.")
    incidence = sp.csc_matrix(incidence)
    n, m = incidence.shape
    adjacency = np.zeros((n, n), dtype=int)
    for j in range(m):
        e = neighbors(j, incidence)
        adjacency[e[0], e[1]] = 1
        adjacency[e[1], e[0]] = 1
    return adjacency


class Kernel:
    """
    Parameters
    ----------
    incidence: :class:`~numpy.ndarray`
        Incidence matrix of the graph to analyze.
    tol: :class:`float`
        Tolerance for approximating zero.

    Attributes
    -----------

    Examples
    --------

    >>> import stochastic_matching.graphs as sm
    >>> paw = sm.Tadpole()
    >>> kernel  = Kernel(paw.incidence)

    The inverse is:

    >>> kernel.inverse
    array([[ 0.5,  0.5, -0.5,  0.5],
           [ 0.5, -0.5,  0.5, -0.5],
           [-0.5,  0.5,  0.5, -0.5],
           [ 0. ,  0. ,  0. ,  1. ]])

    We can check that it is indeed the inverse.

    >>> i = paw.incidence @ kernel.inverse
    >>> clean_zeros(i)
    >>> i
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    Right kernel is trivial:

    >>> kernel.right.shape[0]
    0

    Left kernel is trivial:

    >>> kernel.left.shape[1]
    0

    Graph is bijective:

    >>> kernel.type
    'Bijective'

    As the graph is simple and connected, there a more accurate description of the status:

    >>> status_names_simple_connected[kernel.status]
    'Non-bipartite monocyclic'

    A summary:

    >>> kernel
    Kernels of a graph with 4 nodes and 4 edges.
    Node dimension is 0.
    Edge dimension is 0
    Type: Bijective
    Node kernel:
    []
    Edge kernel:
    []

    Now consider a bipartite version, the banner graph :

    >>> banner = sm.Tadpole(m=4)
    >>> kernel = Kernel(banner.incidence)

    The pseudo-inverse is:

    >>> kernel.inverse
    array([[ 0.35,  0.4 , -0.15, -0.1 ,  0.1 ],
           [ 0.45, -0.2 , -0.05,  0.3 , -0.3 ],
           [-0.15,  0.4 ,  0.35, -0.1 ,  0.1 ],
           [-0.05, -0.2 ,  0.45,  0.3 , -0.3 ],
           [-0.2 ,  0.2 , -0.2 ,  0.2 ,  0.8 ]])

    We can check that it is indeed not exactly the inverse.

    >>> i = banner.incidence @ kernel.inverse
    >>> i
    array([[ 0.8,  0.2, -0.2,  0.2, -0.2],
           [ 0.2,  0.8,  0.2, -0.2,  0.2],
           [-0.2,  0.2,  0.8,  0.2, -0.2],
           [ 0.2, -0.2,  0.2,  0.8,  0.2],
           [-0.2,  0.2, -0.2,  0.2,  0.8]])

    Right kernel is not trivial because of the even cycle:

    >>> kernel.right.shape[0]
    1
    >>> kernel.right # doctest: +SKIP
    array([[ 0.5, -0.5, -0.5,  0.5,  0. ]])

    Left kernel is not trivial because of the bipartite degenerescence:

    >>> kernel.left.shape[1]
    1
    >>> kernel.left
    array([[ 0.4472136],
           [-0.4472136],
           [ 0.4472136],
           [-0.4472136],
           [ 0.4472136]])

    Status is nonjective (not injective nor bijective):

    >>> kernel.type
    'Nonjective'

    As the graph is simple and connected, there a more accurate description of the status:

    >>> status_names_simple_connected[kernel.status]
    'Bipartite with cycle(s)'

    Consider now the diamond graph, surjective (n<m, non bipartite).

    >>> diamond = sm.CycleChain()
    >>> kernel = Kernel(diamond.incidence)

    The inverse is:

    >>> kernel.inverse
    array([[ 0.5 ,  0.25, -0.25,  0.  ],
           [ 0.5 , -0.25,  0.25,  0.  ],
           [-0.5 ,  0.5 ,  0.5 , -0.5 ],
           [ 0.  ,  0.25, -0.25,  0.5 ],
           [ 0.  , -0.25,  0.25,  0.5 ]])

    We can check that it is indeed the inverse.

    >>> i = diamond.incidence @ kernel.inverse
    >>> clean_zeros(i)
    >>> i
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    There is a right kernel:

    >>> kernel.right.shape[0]
    1
    >>> kernel.right
    array([[ 0.5, -0.5,  0. , -0.5,  0.5]])


    The left kernel is trivial:

    >>> kernel.left.shape[1]
    0

    The diamond is surjective-only:

    >>> kernel.type
    'Surjective-only'

    As the graph is simple and connected, there a more accurate description of the status:

    >>> status_names_simple_connected[kernel.status]
    'Non-bipartite polycyclic'

    Consider now a star graph, injective (tree).

    >>> star = sm.Star()
    >>> kernel = Kernel(star.incidence)

    The inverse is:

    >>> kernel.inverse
    array([[ 0.25,  0.75, -0.25, -0.25],
           [ 0.25, -0.25,  0.75, -0.25],
           [ 0.25, -0.25, -0.25,  0.75]])

    We can check that it is indeed the **left** inverse.

    >>> i = kernel.inverse @ star.incidence
    >>> clean_zeros(i)
    >>> i
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    The right kernel is trivial:

    >>> kernel.right.shape[0]
    0

    The left kernel shows the bibartite behavior:

    >>> kernel.left.shape[1]
    1
    >>> kernel.left
    array([[-0.5],
           [ 0.5],
           [ 0.5],
           [ 0.5]])

    The star is injective-only:

    >>> kernel.type
    'Injective-only'

    As the graph is simple and connected, there a more accurate description of the status:

    >>> status_names_simple_connected[kernel.status]
    'Tree'

    Next, a surjective hypergraph:

    >>> clover = sm.Fan()
    >>> kernel = Kernel(clover.incidence)

    Incidence matrix dimensions:

    >>> clover.incidence.shape
    (9, 10)

    The inverse dimensions:

    >>> kernel.inverse.shape
    (10, 9)

    We can check that it is exactly the inverse, because there was no dimensionnality loss.

    >>> i = clover.incidence @ kernel.inverse
    >>> clean_zeros(i)
    >>> i
    array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 1.]])

    Right kernel is 1 dimensional:

    >>> kernel.right.shape[0]
    1

    Left kernel is trivial.

    >>> kernel.left.shape[1]
    0

    Status:

    >>> kernel.type
    'Surjective-only'

    Lastly, observe a *bipartite* hypergraph (in the sense of with non-trivial left kernel).

    >>> clover = sm.Fan(cycle_size=4)
    >>> kernel = Kernel(clover.incidence)

    Incidence matrix dimensions:

    >>> clover.incidence.shape
    (12, 13)

    The inverse dimensions:

    >>> kernel.inverse.shape
    (13, 12)

    We can check that it is not exactly the inverse.

    >>> (clover.incidence @ kernel.inverse)[:4, :4]
    array([[ 0.83333333,  0.16666667, -0.16666667,  0.16666667],
           [ 0.16666667,  0.83333333,  0.16666667, -0.16666667],
           [-0.16666667,  0.16666667,  0.83333333,  0.16666667],
           [ 0.16666667, -0.16666667,  0.16666667,  0.83333333]])

    Right kernel is 3 dimensional:

    >>> kernel.right.shape[0]
    3

    Left kernel is 2-dimensional (this is a change compared to simple graph,
    where the left kernel dimension of a connected component is at most 1).

    >>> kernel.left.shape[1]
    2

    Status:

    >>> kernel.type
    'Nonjective'
    """

    def __init__(self, incidence, tol=1e-10):
        self.tol = tol
        self.right = None
        self.left = None
        self.inverse = None
        self.right_inverse = None
        self.status = None
        self.type = None
        self.fit(incidence)

    def fit(self, incidence):
        n, m = incidence.shape
        min_d = min(n, m)
        u, s, v = np.linalg.svd(incidence)
        clean_zeros(s, tol=self.tol)
        dia = np.zeros((m, n))
        dia[:min_d, :min_d] = np.diag([pseudo_inverse_scalar(e) for e in s])
        ev = np.zeros(m)
        ev[:len(s)] = s
        self.right = v[ev == 0, :]
        eu = np.zeros(n)
        eu[:len(s)] = s
        self.left = u[:, eu == 0]
        self.inverse = v.T @ dia @ u.T
        self.right_inverse = kernel_inverse(self.right)
        clean_zeros(self.inverse, tol=self.tol)
        clean_zeros(self.right, tol=self.tol)
        clean_zeros(self.left, tol=self.tol)
        clean_zeros(self.right_inverse, tol=self.tol)
        injective = self.right.shape[0] == 0
        surjective = self.left.shape[1] == 0
        self.status = (injective, surjective)
        self.type = status_names[self.status]

    def __repr__(self):
        return (f"Kernels of a graph with {self.left.shape[0]} nodes and {self.right.shape[1]} edges.\n"
                f"Node dimension is {self.left.shape[1]}.\n"
                f"Edge dimension is {self.right.shape[0]}\n"
                f"Type: {self.type}\n"
                f"Node kernel:\n"
                f"{self.left}\n"
                f"Edge kernel:\n"
                f"{self.right}")


def kernel_inverse(kernel):
    """
    Parameters
    ----------
    kernel: :class:`numpy.ndarray`
        Matrix of kernel vectors (not necessarily orthogonal) of shape dXm.

    Returns
    -------
    :class:`numpy.ndarray`
        The `reverse` matrix dXd that allows to transform inner product with kernel to kernel coordinates.

    Examples
    --------

    When the kernel basis is orthogonal,
    it returns the diagonal matrix with the inverse of the squared norm of the vectors.
    For example:

    >>> edge_kernel = np.array([[ 0,  0,  1, -1, -1,  1,  0,  0],
    ...       [ 1, -1,  0, -1,  1,  0, -1,  1]])
    >>> inverse = kernel_inverse(edge_kernel)
    >>> inverse
    array([[0.25      , 0.        ],
           [0.        , 0.16666667]])

    Here the kernel basis is orthogonal,
    so you just have a diagonal matrix with the inverse square of the vector norms.

    Consider [-1, 3] in kernel coordinates. In edge coordinates, it corresponds to:

    >>> edges = np.array([-1, 3]) @ edge_kernel
    >>> edges
    array([ 3, -3, -1, -2,  4, -1, -3,  3])

    One can project the edges on the kernel.

    >>> projection = edge_kernel @ edges
    >>> projection
    array([-4, 18])

    The kernel inverse allows to rectify the projection in kernel coordinates.

    >>> projection @ inverse
    array([-1.,  3.])

    If the kernel basis is not orthogonal, it returns somethings more complex.

    >>> edge_kernel = np.array([[ 1, -1,  1, -2,  0,  1, -1,  1],
    ...    [ 0,  0, -1,  1,  1, -1,  0,  0]])
    >>> inverse = kernel_inverse(edge_kernel)
    >>> inverse
    array([[0.16666667, 0.16666667],
           [0.16666667, 0.41666667]])

    The inverse is more complex because the basis is not orthogonal.

    >>> edges = np.array([2, -1]) @ edge_kernel
    >>> edges
    array([ 2, -2,  3, -5, -1,  3, -2,  2])

    >>> projection = edge_kernel @ edges
    >>> projection
    array([ 24, -12])

    >>> projection @ inverse
    array([ 2., -1.])
    """
    return np.linalg.inv(np.inner(kernel, kernel))


def simple_right_kernel(edge_kernel, seeds):
    """
    Parameters
    ----------
    edge_kernel: :class:`~numpy.ndarray`
        Right kernel (i.e. edges kernel) of a simple graph.
    seeds: :class:`list` of :class:`int`
        Seed edges of the kernel space. Valid seeds can be obtained from
        :meth:`~stochastic_matching.analysis.connected_components`.
        Cf https://hal.archives-ouvertes.fr/hal-03502084.

    Returns
    -------
    :class:`~numpy.ndarray`
        The kernel expressed as elements from the cycle space (even cycles and kayak paddles).

    Examples
    --------

    Start with the co-domino.

    >>> import stochastic_matching.graphs as sm

    Default decomposition is a square and an hex.

    >>> right = sm.Codomino().kernel.right
    >>> right
    array([[ 0,  0,  1, -1, -1,  1,  0,  0],
           [ 1, -1,  0, -1,  1,  0, -1,  1]])

    A second possible decomposition with a kayak paddle and a square.

    >>> simple_right_kernel(right, [0, 4])
    array([[ 1, -1,  1, -2,  0,  1, -1,  1],
           [ 0,  0, -1,  1,  1, -1,  0,  0]])

    Another example with the pyramid.

    >>> right = sm.Pyramid().kernel.right

    Default decomposition: cycles of length 8, 10, and 6.

    >>> right
    array([[ 0,  0,  1, -1, -1,  1,  0, -1,  1, -1,  1,  0,  0],
           [-1,  1,  0,  1, -1,  1,  0, -1, -1,  1,  0, -1,  1],
           [ 1, -1,  0, -1,  1, -1,  1,  0,  0,  0,  0,  0,  0]])

    Second decomposition: two cycles of length 6, one of length 8.

    >>> simple_right_kernel(right, [0, 12, 2])
    array([[ 1, -1,  0, -1,  1, -1,  1,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0, -1,  1],
           [ 0,  0,  1, -1, -1,  1,  0, -1,  1, -1,  1,  0,  0]])

    Another decomposition: two cycles of length 6 and a kayak paddle :math:`KP_{3, 3, 3}`.

    >>> simple_right_kernel(right, [5, 7, 2])
    array([[-1,  1,  0,  1, -1,  1, -1,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0, -1,  1,  1, -1,  0,  1, -1],
           [ 1, -1,  1, -2,  0,  0,  0,  0,  2, -2,  1,  1, -1]])
    """
    return np.round(np.linalg.inv(edge_kernel[:, seeds]) @ edge_kernel).astype(int)


def simple_left_kernel(left):
    """

    Parameters
    ----------
    left: :class:`~numpy.ndarray`
        Left kernel (i.e. nodes kernel) of a simple graph, corresponding to bipartite components

    Returns
    -------
    :class:`~numpy.ndarray`
        The kernel with infinite-norm renormalization.

    Examples
    --------

    By default the kernel vector are 2-normalized.

    >>> import stochastic_matching.graphs as sm
    >>> sample = sm.concatenate([sm.Cycle(4), sm.Star(5)], 0)
    >>> raw_kernel = Kernel(sample.incidence)
    >>> raw_kernel.left
    array([[ 0.5      ,  0.       ],
           [-0.5      ,  0.       ],
           [ 0.5      ,  0.       ],
           [-0.5      ,  0.       ],
           [ 0.       , -0.4472136],
           [ 0.       ,  0.4472136],
           [ 0.       ,  0.4472136],
           [ 0.       ,  0.4472136],
           [ 0.       ,  0.4472136]])

    `simple_left_kernel` adjusts the values to {-1, 0, 1}

    >>> simple_left_kernel(raw_kernel.left)
    array([[ 1,  0],
           [-1,  0],
           [ 1,  0],
           [-1,  0],
           [ 0, -1],
           [ 0,  1],
           [ 0,  1],
           [ 0,  1],
           [ 0,  1]])
    """
    return np.around(left/np.max(left[:], axis=0)).astype(int)


def traversal(model):
    """
    Using graph traversal, splits the graph into its connected components as a list of sets of nodes and edges.
    If the graph is simple, additional information obtained by the traversal are provided.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model with graph to decompose.

    Returns
    -------
    :class:`list` of :class:`dict`
        The list of connected components.
        If the graph is not simple, each connected component contains its sets of nodes and edges.
        If the graph is simple, each component also contains a set of spanning edges, a pivot edge that
        makes the spanner bijective (if any), a set set of edges that can seed the kernel space, and the type
        of the connected component.

    Examples
    ---------

    For simple graphs, the method provides a lot of information on each connected component.

    >>> import stochastic_matching.graphs as sm
    >>> tutti = sm.concatenate([sm.Cycle(4), sm.Complete(4), sm.CycleChain(), sm.Tadpole(), sm.Star()], 0)
    >>> traversal(tutti) # doctest: +NORMALIZE_WHITESPACE
    [{'nodes': {0, 1, 2, 3}, 'edges': {0, 1, 2, 3},
    'spanner': {0, 1, 2}, 'pivot': False, 'seeds': {3},
    'type': 'Bipartite with cycle(s)'},
    {'nodes': {4, 5, 6, 7}, 'edges': {4, 5, 6, 7, 8, 9},
    'spanner': {4, 5, 6}, 'pivot': 8, 'seeds': {9, 7},
    'type': 'Non-bipartite polycyclic'},
    {'nodes': {8, 9, 10, 11}, 'edges': {10, 11, 12, 13, 14},
    'spanner': {10, 11, 13}, 'pivot': 12, 'seeds': {14},
    'type': 'Non-bipartite polycyclic'},
    {'nodes': {12, 13, 14, 15}, 'edges': {16, 17, 18, 15},
    'spanner': {16, 18, 15}, 'pivot': 17, 'seeds': set(),
    'type': 'Non-bipartite monocyclic'},
    {'nodes': {16, 17, 18, 19}, 'edges': {19, 20, 21},
    'spanner': {19, 20, 21}, 'pivot': False, 'seeds': set(),
    'type': 'Tree'}]

    These informations make the analysis worthy even in the cases where the graph is connected.

    >>> traversal(sm.Pyramid()) # doctest: +NORMALIZE_WHITESPACE
    [{'nodes': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    'edges': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    'spanner': {0, 1, 3, 4, 5, 7, 8, 9, 11},
    'pivot': 2, 'seeds': {10, 12, 6},
    'type': 'Non-bipartite polycyclic'}]

    If the graph is treated as hypergraph, a lot less information is available.

    >>> tutti.adjacency = None
    >>> traversal(tutti) # doctest: +NORMALIZE_WHITESPACE
    [{'nodes': {0, 1, 2, 3}, 'edges': {0, 1, 2, 3}},
    {'nodes': {4, 5, 6, 7}, 'edges': {4, 5, 6, 7, 8, 9}},
    {'nodes': {8, 9, 10, 11}, 'edges': {10, 11, 12, 13, 14}},
    {'nodes': {12, 13, 14, 15}, 'edges': {16, 17, 18, 15}},
    {'nodes': {16, 17, 18, 19}, 'edges': {19, 20, 21}}]
    """
    simple = model.adjacency is not None
    n, m = model.n, model.m
    unknown_nodes = {i for i in range(n)}
    if simple:
        spin = np.ones(n, dtype=bool)  # Simple
    res = []
    while unknown_nodes:
        buffer = {unknown_nodes.pop()}
        current_nodes = set()
        current_edges = set()
        if simple:
            current_spanner = set()
        while buffer:
            i = buffer.pop()
            current_nodes.add(i)
            edges = neighbors(i, model.incidence_csr)
            for edge in edges:
                if edge in current_edges:
                    continue
                for j in neighbors(edge, model.incidence_csc):
                    if j in unknown_nodes:
                        buffer.add(j)
                        unknown_nodes.remove(j)
                        if simple:
                            spin[j] = not spin[i]  # Simple
                            current_spanner.add(edge)
                current_edges.add(edge)
        cc = {'nodes': current_nodes, 'edges': current_edges}
        if simple:
            cc['spanner'] = current_spanner  # Simple
            free_edges = current_edges - current_spanner
            for edge in free_edges:
                pair = neighbors(edge, model.incidence_csc)
                if spin[pair[0]] == spin[pair[1]]:
                    cc['pivot'] = edge
                    free_edges.discard(edge)
                    break
            else:
                cc['pivot'] = False
            cc['seeds'] = free_edges
            injective = len(free_edges) == 0
            surjective = cc['pivot'] is not False
            cc['type'] = status_names_simple_connected[(injective, surjective)]
        res.append(cc)
    return res


class Model:
    """
    Main class to manipulate stochatic matching models.

    Parameters
    ----------
    incidence: :class:`~numpy.ndarray`, optional
        Incidence matrix. If used, the graph will be considered as a hypergraph.
    adjacency: :class:`~numpy.ndarray`, optional
        Adjacency matrix. If used, the graph will be considered as a simple graph.
    rates: :class:`~numpy.ndarray` or :class:`list` or :class:`str`, optional
        Arrival rates. You can use a specific rate vector or list.
        You can use `uniform` or `proportional` for uniform or degree-proportional allocation.
        Default to `proportional`, which makes the problem stabilizable if the graph is bijective.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)
    tol: :class:`float`, optional
        Values of absolute value lower than `tol` are set to 0.

    Examples
    --------

    The following examples are about stability:

    Is a triangle that checks triangular inequality stable?

    >>> import stochastic_matching.graphs as sm
    >>> triangle = sm.Cycle(rates="uniform")
    >>> triangle.stabilizable
    True

    >>> triangle.kernel.type
    'Bijective'

    We can look at the base flow (based on Moore-Penrose inverse by default).

    >>> triangle.base_flow
    array([0.5, 0.5, 0.5])

    As the graph is bijective, all optimizations will yield the same flow.

    >>> triangle.incompressible_flow()
    array([0.5, 0.5, 0.5])

    >>> triangle.optimize_edge(0)
    array([0.5, 0.5, 0.5])

    What if the triangular inequality does not hold?

    >>> triangle.rates = [1, 3, 1]
    >>> triangle.stabilizable
    False

    We can look at the base flow (based on Moore-Penrose inverse).

    >>> triangle.base_flow
    array([ 1.5, -0.5,  1.5])

    Now a bipartite example.

    >>> banner = sm.Tadpole(m=4, rates='proportional')

    Notice that we have a perfectly working solution with respect to conservation law.

    >>> banner.base_flow
    array([1., 1., 1., 1., 1.])

    However, the left kernel is not trivial.

    >>> banner.kernel.left
    array([[ 1],
           [-1],
           [ 1],
           [-1],
           [ 1]])

    As a consequence, stability is False.

    >>> banner.stabilizable
    False

    >>> banner.kernel.type
    'Nonjective'

    Note that the base flow can be negative even if there is a positive solution.

    >>> diamond = sm.CycleChain(rates=[5, 6, 2, 1])
    >>> diamond.base_flow
    array([ 3.5,  1.5,  1. ,  1.5, -0.5])
    >>> diamond.stabilizable
    True
    >>> diamond.maximin.round(decimals=6)
    array([4.5, 0.5, 1. , 0.5, 0.5])

    >>> diamond.incompressible_flow()
    array([4., 0., 1., 0., 0.])

    >>> diamond.kernel.type
    'Surjective-only'
    """
    name = "Generic"

    def __init__(self, incidence=None, adjacency=None, rates=None, names=None, tol=1e-7):
        self.tol = tol
        self.base_flow = None
        self.__seeds = None
        self.incidence = incidence
        self.adjacency = adjacency
        self.rates = rates
        self.names = names
        self.simulator = None
        self.simulation = None

    @property
    def names(self):
        """
        :class:`list` of :class:`str`: list of node names (e.g. for display).

        If set to "alpha", automatic alphabetic labeling is used.
        If set to None, numeric labeling is used.
        """
        return self.__names

    @names.setter
    def names(self, names):
        if type(names) == str and names == "alpha":
            self.__names = CharMaker()
        else:
            self.__names = names

    @property
    def incidence(self):
        """
        :class:`~numpy.ndarray`: Incidence matrix of the graph.

        Setting incidence treats the graph as hypergraph.
        """
        return self._incidence.toarray().astype(int)

    @property
    def incidence_csr(self):
        """
        :class:`~scipy.sparse.csr_matrix`: Incidence matrix of the graph in csr format.
        """
        return self._incidence

    @property
    def incidence_csc(self):
        """
        :class:`~scipy.sparse.csc_matrix`: Incidence matrix of the graph in csc format.
        """
        return self._co_incidence

    @incidence.setter
    def incidence(self, incidence):
        if incidence is None:
            self._incidence = None
        else:
            if type(incidence) == list:
                incidence = np.array(incidence).astype(int)
            self._incidence = sp.csr_matrix(incidence)
            self._co_incidence = sp.csc_matrix(incidence)
            self._adjacency = None

    @property
    def adjacency(self):
        """
        :class:`~numpy.ndarray`: Adjacency matrix of the graph.

        Setting adjacency treats the graph as simple.
        """
        return self._adjacency

    @adjacency.setter
    def adjacency(self, adjacency):
        if adjacency is not None:
            adjacency = np.array(adjacency)
            incidence = adjacency_to_incidence(adjacency)
            self._incidence = sp.csr_matrix(incidence)
            self._co_incidence = sp.csc_matrix(incidence)
        self._adjacency = adjacency

    @property
    def n(self):
        """
        :class:`int`: Number of nodes.
        """
        return self._incidence.shape[0]

    @property
    def m(self):
        """
        :class:`int`: Number of edges.
        """
        return self._incidence.shape[1]

    @cached_property
    def degree(self):
        """
        :class:`~numpy.ndarray`: Degree vector.
        """
        return self._incidence @ np.ones(self.m)

    @cached_property
    def kernel(self):
        """
        :class:`~stochastic_matching.model.Kernel`: Description of the kernels of the incidence.
        """
        kernel = Kernel(self.incidence)
        if self.adjacency is not None:
            self.__seeds = [i for c in self.connected_components for i in c['seeds']]
            kernel.right = simple_right_kernel(kernel.right, self.__seeds)
            kernel.right_inverse = kernel_inverse(kernel.right)
            kernel.left = simple_left_kernel(kernel.left)
        return kernel

    @cached_property
    def connected_components(self):
        """
        :class:`list` of :class:`dict`: description of the connected components of the graph.
        """
        return traversal(self)

    @property
    def seeds(self):
        """
        :class:`list` of :class:`int`: edges that induce a description of the (right) kernel as cycles and paddles.

        Changing the seeds modifies the right kernel accordingly. Only available on simple graphs.
        """
        if self.adjacency is not None:
            return self.__seeds

    @seeds.setter
    def seeds(self, seeds):
        if self.adjacency is not None:
            self.kernel.right = simple_right_kernel(self.kernel.right, seeds)
            self.kernel.right_inverse = kernel_inverse(self.kernel.right)
            self.__seeds = seeds

    @property
    def rates(self):
        """
        :class:`~numpy.ndarray`: vector of arrival rates.

        You can use `uniform` or `proportional` for uniform or degree-proportional allocation.
        Default to `proportional`, which makes the problem stabilizable if the graph is bijective.
        """
        return self.__rates

    @rates.setter
    def rates(self, mu):
        if mu is None:
            mu = self.degree
        if mu is not None:
            if isinstance(mu, str):
                if mu == 'uniform':
                    mu = np.ones(self.n)
                else:
                    mu = self.degree
            else:
                mu = np.array(mu)
        self.__rates = mu
        self.__moore_penrose = None
        self.__maximin = None
        self.__vertices = None
        self.base_flow = self.moore_penrose

    @property
    def moore_penrose(self):
        """
        :class:`~numpy.ndarray`: Solution of the conservation equation obtained using the Moore-Penrose inverse.
        """
        if self.__moore_penrose is None:
            self.__moore_penrose = self.kernel.inverse @ self.rates
            clean_zeros(self.__moore_penrose, tol=self.tol)
        return self.__moore_penrose

    def edge_to_kernel(self, edge):
        """
        Parameters
        ----------
        edge: :class:`~numpy.ndarray`
            A flow vector in edge coordinates.

        Returns
        -------
        :class:`~numpy.ndarray`
            The same flow vector in kernel coordinates, based on the current base flow and right kernel.

        Examples
        --------

        Consider the codomino graph with a kernel with a kayak paddle.

        >>> import stochastic_matching.graphs as sm
        >>> codomino = sm.Codomino(rates = [3, 12, 3, 3, 12, 3])

        Default seeds of the codomino:

        >>> codomino.seeds
        [5, 7]

        Let use other seeds.

        >>> codomino.seeds = [0, 4]
        >>> codomino.kernel.right  # doctest: +NORMALIZE_WHITESPACE
        array([[ 1, -1,  1, -2,  0,  1, -1,  1],
           [ 0,  0, -1,  1,  1, -1,  0,  0]])

        Consider the Moore-Penrose flow and the maximin flow.

        >>> codomino.moore_penrose
        array([3., 0., 3., 6., 0., 3., 0., 3.])

        >>> codomino.maximin
        array([2., 1., 1., 9., 1., 1., 1., 2.])

        As the Moore-Penrose is the default base flow, its coordinates are obviously null.

        >>> codomino.edge_to_kernel(codomino.moore_penrose)
        array([0., 0.])

        As for maximin, one can check that the following kernel coordinates transform Moore-Penrose into it:

        >>> codomino.edge_to_kernel(codomino.maximin)
        array([-1.,  1.])

        If we change the base flow to maximin, we will see the coordinates shifted by (1, -1):

        >>> codomino.base_flow = codomino.maximin
        >>> codomino.edge_to_kernel(codomino.moore_penrose)
        array([ 1., -1.])
        >>> codomino.edge_to_kernel(codomino.maximin)
        array([0., 0.])
        """
        res = (self.kernel.right @ (edge - self.base_flow)) @ self.kernel.right_inverse
        clean_zeros(res, tol=self.tol)
        return res

    def kernel_to_edge(self, alpha):
        """
        Parameters
        ----------
        alpha: :class:`~numpy.ndarray` ot :class:`list`
            A flow vector in kernel coordinates.

        Returns
        -------
        :class:`~numpy.ndarray`
            The same flow vector in edge coordinates, based on the current base flow and right kernel.

        Examples
        --------

        Consider the codomino graph with a kernel with a kayak paddle.

        >>> import stochastic_matching.graphs as sm
        >>> codomino = sm.Codomino(rates=[3, 12, 3, 3, 12, 3])
        >>> codomino.kernel.right # doctest: +NORMALIZE_WHITESPACE
        array([[ 0,  0,  1, -1, -1,  1,  0,  0],
           [ 1, -1,  0, -1,  1,  0, -1,  1]])

        Consider the Moore-Penrose and the maximin flows.

        >>> codomino.moore_penrose
        array([3., 0., 3., 6., 0., 3., 0., 3.])

        >>> codomino.maximin
        array([2., 1., 1., 9., 1., 1., 1., 2.])

        As the Moore-Penrose inverse is the base flow, it is (0, 0) in kernel coordinates.

        >>> codomino.kernel_to_edge([0, 0])
        array([3., 0., 3., 6., 0., 3., 0., 3.])

        As for maximin, (-2, -1) seems to be its kernel coordinates.

        >>> codomino.kernel_to_edge([-2, -1])
        array([2., 1., 1., 9., 1., 1., 1., 2.])
        >>> np.allclose(codomino.kernel_to_edge([-2, -1]), codomino.maximin)
        True

        If we change the kernel space, the kernel coordinates change as well.

        >>> codomino.seeds = [0, 4]
        >>> codomino.kernel.right # doctest: +NORMALIZE_WHITESPACE
        array([[ 1, -1,  1, -2,  0,  1, -1,  1],
           [ 0,  0, -1,  1,  1, -1,  0,  0]])
        >>> codomino.kernel_to_edge([0, 0])
        array([3., 0., 3., 6., 0., 3., 0., 3.])

        >>> codomino.kernel_to_edge([-2, -1])
        array([ 1.,  2.,  2.,  9., -1.,  2.,  2.,  1.])
        >>> np.allclose(codomino.kernel_to_edge([-2, -1]), codomino.maximin)
        False
        """
        res = (alpha @ self.kernel.right) + self.base_flow
        clean_zeros(res, tol=self.tol)
        return res

    @property
    def maximin(self):
        """
        :class:`~numpy.ndarray`: solution of the conservation equation that maximizes the minimal flow over all edges.
        """
        if self.__maximin is None:
            d, m = self.kernel.right.shape
            if d == 0:
                self.__maximin = self.moore_penrose
            else:
                # Better in theory but better precision with the legacy approach
                # c = np.zeros(m + 1)
                # c[-1] = -1
                # a_ub = sp.hstack([-sp.identity(m), sp.csr_matrix(np.ones((m, 1)))])
                # b_ub = np.zeros(m)
                # a_eq = sp.csr_matrix((self.incidence_csr.data,
                #                    self.incidence_csr.indices,
                #                    self.incidence_csr.indptr), shape=(self.n, m + 1))
                # b_eq = self.rates
                # optimizer = linprog(c=c,
                #                     A_ub=a_ub,
                #                     b_ub=b_ub,
                #                     A_eq=a_eq,
                #                     b_eq=b_eq,
                #                     bounds=[(None, None)] * (m + 1),
                #                     options={'sparse': True}
                #                     )
                # flow = optimizer.x[:-1]

                c = np.zeros(d + 1)
                c[d] = 1
                a_ub = -np.vstack([self.kernel.right, np.ones(m)]).T
                optimizer = linprog(c=c,
                                    A_ub=a_ub,
                                    b_ub=self.moore_penrose,
                                    bounds=[(None, None)] * (d + 1)
                                    )
                flow = optimizer.slack - optimizer.x[-1]
                clean_zeros(flow, tol=self.tol)
                self.__maximin = flow
        return self.__maximin

    def optimize_rates(self, weights):
        """
        Tries to find a positive solution that minimizes/maximizes a given edge.

        Parameters
        ----------
        weights: :class:`~numpy.ndarray` or :class:`list`
            Rewards associated to each edge.

        Returns
        -------
        :class:`~numpy.ndarray`
            Optimized flow that maximize the total reward.

        Examples
        --------

        >>> import stochastic_matching as sm
        >>> diamond = sm.CycleChain(rates=[4, 5, 2, 1])

        Optimize with the first edge that provides no reward.

        >>> diamond.optimize_rates([0, 1, 1, 1, 1])
        array([3., 1., 1., 1., 0.])

        Optimize with the first edge that provides more reward.

        >>> diamond.optimize_rates([2, 1, 1, 1, 1])
        array([4., 0., 1., 0., 1.])

        On bijective graphs, the method directly returns the unique solution.

        >>> paw = sm.Tadpole()
        >>> paw.optimize_rates([6, 3, 1, 2])
        array([1., 1., 1., 1.])
        """
        d, m = self.kernel.right.shape
        if d == 0:
            return self.moore_penrose
        else:
            weights = np.array(weights)
            c = - self.kernel.right @ weights
            optimizer = linprog(c=c,
                                A_ub=-self.kernel.right.T,
                                b_ub=self.moore_penrose,
                                bounds=[(None, None)] * d
                                )
            clean_zeros(optimizer.slack, tol=self.tol)
            return optimizer.slack

    def optimize_edge(self, edge, sign=1):
        """
        Tries to find a positive solution that minimizes/maximizes a given edge.

        Parameters
        ----------
        edge: :class:`int`
            Edge to optimize.
        sign: :class:`int`
            Use 1 to maximize, -1 to minimize.

        Returns
        -------
        :class:`~numpy.ndarray`
            Optimized flow.
        """
        d, m = self.kernel.right.shape
        if d == 0:
            return self.moore_penrose
        else:
            weights = np.zeros(self.m)
            weights[edge] = sign
            return self.optimize_rates(weights)

    def incompressible_flow(self):
        """
        Finds the minimal flow that must pass through each edge. This is currently done in a *brute force* way
        by minimizing every edges.

        Returns
        -------
        :class:`~numpy.ndarray`
            Unavoidable flow.
        """
        d, m = self.kernel.right.shape
        if d == 0:
            return self.base_flow
        else:
            flow = np.zeros(m)
            for edge in range(m):
                flow[edge] = self.optimize_edge(edge, -1)[edge]
            clean_zeros(flow, tol=self.tol)
            return flow

    @property
    def stabilizable(self):
        """
        :class:`bool`: Is the model stabilizable, i.e is it bijective with a positive solution of the conservation law?
        """
        return np.all(self.maximin > 0) and self.kernel.status[1]

    @property
    def vertices(self):
        """
        :class:`list`: list of vertices

        Examples
        --------

        >>> import stochastic_matching.graphs as sm
        >>> paw = sm.Tadpole()
        >>> paw.vertices  # doctest: +NORMALIZE_WHITESPACE
        [{'kernel_coordinates': None, 'edge_coordinates': array([1., 1., 1., 1.]),
          'null_edges': [], 'bijective': True}]

        Note that vertices is soft-cached: as long as the graph and rates do not change they are not re-computed.

        >>> paw.vertices  # doctest: +NORMALIZE_WHITESPACE
        [{'kernel_coordinates': None, 'edge_coordinates': array([1., 1., 1., 1.]),
          'null_edges': [], 'bijective': True}]


        >>> star = sm.Star()
        >>> star.vertices  # doctest: +NORMALIZE_WHITESPACE
        [{'kernel_coordinates': None, 'edge_coordinates': array([1., 1., 1.]),
          'null_edges': [], 'bijective': False}]

        >>> cycle = sm.Cycle(4, rates=[3, 3, 3, 1])
        >>> cycle.vertices  # doctest: +NORMALIZE_WHITESPACE
        [{'kernel_coordinates': array([-0.75]),
          'edge_coordinates': array([1. , 1.5, 2.5, 0. ]),
          'null_edges': [3], 'bijective': False},
         {'kernel_coordinates': array([0.75]), 'edge_coordinates': array([2.5, 0. , 1. , 1.5]),
          'null_edges': [1], 'bijective': False}]

        >>> diamond = sm.CycleChain(rates=[3, 3, 3, 1])
        >>> diamond.vertices  # doctest: +NORMALIZE_WHITESPACE
        [{'kernel_coordinates': array([-0.5]), 'edge_coordinates': array([1., 2., 1., 1., 0.]),
          'null_edges': [4], 'bijective': True},
         {'kernel_coordinates': array([0.5]), 'edge_coordinates': array([2., 1., 1., 0., 1.]),
         'null_edges': [3], 'bijective': True}]

        >>> diamond.rates = [2, 3, 2, 1]
        >>> diamond.vertices  # doctest: +NORMALIZE_WHITESPACE
        [{'kernel_coordinates': array([-0.25]), 'edge_coordinates': array([1., 1., 1., 1., 0.]),
          'null_edges': [4], 'bijective': True},
         {'kernel_coordinates': array([0.75]), 'edge_coordinates': array([2., 0., 1., 0., 1.]),
          'null_edges': [1, 3], 'bijective': False}]

        >>> diamond.rates = [1, 3, 1, 1]
        >>> diamond.vertices  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: The matching model admits no positive solution.

        >>> diamond.rates = None
        >>> diamond.vertices  # doctest: +NORMALIZE_WHITESPACE
        [{'kernel_coordinates': array([-1.]), 'edge_coordinates': array([0., 2., 1., 2., 0.]),
          'null_edges': [0, 4], 'bijective': False},
         {'kernel_coordinates': array([1.]), 'edge_coordinates': array([2., 0., 1., 0., 2.]),
          'null_edges': [1, 3], 'bijective': False}]

        >>> codomino = sm.Codomino()
        >>> codomino.rates = [4, 5, 5, 3, 3, 2]
        >>> codomino.seeds = [1, 2]
        >>> codomino.base_flow = codomino.maximin
        >>> sorted(codomino.vertices, key=str)  # doctest: +NORMALIZE_WHITESPACE
        [{'kernel_coordinates': array([ 1., -1.]), 'edge_coordinates': array([1., 3., 1., 3., 1., 0., 2., 0.]),
        'null_edges': [5, 7], 'bijective': True},
        {'kernel_coordinates': array([-1.,  0.]), 'edge_coordinates': array([3., 1., 2., 0., 2., 1., 0., 2.]),
        'null_edges': [3, 6], 'bijective': True},
        {'kernel_coordinates': array([-1., -1.]), 'edge_coordinates': array([3., 1., 1., 1., 3., 0., 0., 2.]),
        'null_edges': [5, 6], 'bijective': True},
        {'kernel_coordinates': array([0., 1.]), 'edge_coordinates': array([2., 2., 3., 0., 0., 2., 1., 1.]),
        'null_edges': [3, 4], 'bijective': True},
        {'kernel_coordinates': array([1., 0.]), 'edge_coordinates': array([1., 3., 2., 2., 0., 1., 2., 0.]),
        'null_edges': [4, 7], 'bijective': True}]

        >>> codomino.rates = [2, 4, 2, 2, 4, 2]
        >>> codomino.base_flow = np.array([1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0])
        >>> sorted(codomino.vertices, key=str)  # doctest: +NORMALIZE_WHITESPACE
        [{'kernel_coordinates': array([ 1., -1.]), 'edge_coordinates': array([0., 2., 0., 4., 0., 0., 2., 0.]),
        'null_edges': [0, 2, 4, 5, 7], 'bijective': False},
        {'kernel_coordinates': array([-1.,  1.]), 'edge_coordinates': array([2., 0., 2., 0., 0., 2., 0., 2.]),
        'null_edges': [1, 3, 4, 6], 'bijective': False},
        {'kernel_coordinates': array([-1., -1.]), 'edge_coordinates': array([2., 0., 0., 2., 2., 0., 0., 2.]),
        'null_edges': [1, 2, 5, 6], 'bijective': False}]

        >>> pyramid = sm.Pyramid(rates=[4, 3, 3, 3, 6, 6, 3, 4, 4, 4])
        >>> pyramid.seeds = [0, 12, 2]
        >>> pyramid.base_flow = pyramid.kernel_to_edge([1/6, 1/6, 1/6] )
        >>> sorted([(v['kernel_coordinates'], v['bijective']) for v in pyramid.vertices], key=str) # doctest: +NORMALIZE_WHITESPACE
        [(array([ 1., -1.,  0.]), True),
         (array([-1.,  1.,  0.]), True),
         (array([-1., -1.,  0.]), True),
         (array([0., 0., 1.]), False),
         (array([1., 1., 0.]), True)]
        """
        if self.__vertices is not None:
            return self.__vertices
        if not np.all(self.maximin > 0):
            raise ValueError("The matching model admits no positive solution.")
        d, m = self.kernel.right.shape
        if d == 0:
            dico = {'kernel_coordinates': None, 'edge_coordinates': self.maximin,
                    'null_edges': [i for i in range(m) if self.maximin[i]==0]}
            dico['bijective'] = ((self.m - self.n) == len(dico['null_edges']))
            res = [dico]
        else:
            if d > 1:
                halfspaces = np.hstack([-self.kernel.right.T, -self.base_flow.reshape(self.m, 1)])
                interior_point = self.edge_to_kernel(self.maximin)
                hs = HalfspaceIntersection(halfspaces, interior_point)
                verts = hs.intersections
                clean_zeros(verts)
                verts = np.unique(verts, axis=0)
            else:
                verts = [None, None]
                for e in range(m):
                    spin = self.kernel.right[0, e]
                    if spin == 0:
                        continue
                    v = -self.base_flow[e] / spin
                    if spin > 0:
                        if verts[0] is None:
                            verts[0] = v
                        else:
                            verts[0] = max(v, verts[0])
                    else:
                        if verts[1] is None:
                            verts[1] = v
                        else:
                            verts[1] = min(v, verts[1])
                verts = np.array(verts).reshape((2, 1))
            v = verts.shape[0]
            res = [dict() for _ in range(v)]
            for i, dico in enumerate(res):
                dico['kernel_coordinates'] = verts[i, :]
                dico['edge_coordinates'] = self.kernel_to_edge(dico['kernel_coordinates'])
                clean_zeros(dico['edge_coordinates'], tol=self.tol)
                dico['null_edges'] = [ i for i in range(m) if dico['edge_coordinates'][i] == 0 ]
                dico['bijective'] = ((self.m - self.n) == len(dico['null_edges']))
        self.__vertices = res
        return res

    def show(self, **kwargs):
        """
        Shows the model. See :meth:`~stochastic_matching.display.show` for details.

        Parameters
        ----------
        **kwargs
            See :meth:`~stochastic_matching.display.show` for details.

        Returns
        -------
        :class:`~IPython.display.HTML`

        Examples
        --------

        >>> import stochastic_matching.graphs as sm
        >>> pyramid = sm.Pyramid()
        >>> pyramid.show()
        <IPython.core.display.HTML object>
        """
        show(self, **kwargs)

    def show_graph(self, **kwargs):
        """
        Shows the graph of the model (with node names and no value on edges by default).

        Parameters
        ----------
        **kwargs
            See :meth:`~stochastic_matching.display.show` for details.

        Returns
        -------
        :class:`~IPython.display.HTML`

        Examples
        --------

        >>> import stochastic_matching.graphs as sm
        >>> pyramid = sm.Pyramid()
        >>> pyramid.show_graph()
        <IPython.core.display.HTML object>
        """
        default = {'disp_flow': False, 'disp_rates': False}
        show(self, **{**default, **kwargs})

    def show_flow(self, **kwargs):
        """
        Shows the model (with check on conservation law and edge positivity).

        Parameters
        ----------
        **kwargs
            See :meth:`~stochastic_matching.display.show` for details.

        Returns
        -------
        :class:`~IPython.display.HTML`

        Examples
        --------

        >>> import stochastic_matching.graphs as sm
        >>> pyramid = sm.Pyramid()
        >>> pyramid.show_flow()
        <IPython.core.display.HTML object>
        """
        default = {'check_flow': True}
        show(self, **{**default, **kwargs})

    def show_kernel(self, **kwargs):
        """
        Shows the kernel basis.

        Parameters
        ----------
        **kwargs
            See :meth:`~stochastic_matching.display.show` for details.

        Returns
        -------
        :class:`~IPython.display.HTML`

        Examples
        --------

        >>> import stochastic_matching.graphs as sm
        >>> pyramid = sm.Pyramid()
        >>> pyramid.show_kernel()
        <IPython.core.display.HTML object>
        """
        default = {'disp_kernel': True, 'disp_flow': False, 'disp_zero': False}
        show(self, **{**default, **kwargs})

    def show_vertex(self, i, **kwargs):
        """
        Shows the vertex of indice *i*. See :meth:`~stochastic_matching.display.show` for details.

        Parameters
        ----------
        i: :class:`int`
            indice of the vertex.
        **kwargs
            See :meth:`~stochastic_matching.display.show` for details.

        Returns
        -------
        :class:`~IPython.display.HTML`

        Examples
        --------

        >>> import stochastic_matching.graphs as sm
        >>> pyramid = sm.Pyramid()
        >>> pyramid.show_vertex(2)
        <IPython.core.display.HTML object>
        """
        flow = self.vertices[i]['edge_coordinates']
        default = {'flow': flow, 'disp_zero': False, 'check_flow': True}
        show(self, **{**default, **kwargs})

    def run(self, simulator, **kwargs):
        """
        All-in-one instantiate and run simulation.

        Parameters
        ----------
        simulator: :class:`str` or :class:`~stochastic_matching.simulator.generic.Simulator`
            Type of simulator to instantiate.
        **kwargs
            Arguments to pass to the simulator.

        Returns
        -------
        bool
            Success of simulation.

        Examples
        --------

        Let start with a working triangle and a greedy simulator.

        >>> import stochastic_matching.graphs as sm
        >>> triangle = sm.Cycle(rates=[3, 4, 5])
        >>> triangle.base_flow
        array([1., 2., 3.])
        >>> triangle.run('random_node', seed=42, number_events=20000)
        True
        >>> triangle.simulation
        array([1.044 , 2.0352, 2.9202])

        A ill diamond graph (simulation ends before completion due to drift).

        Note that the drift is slow, so if the number of steps is low the simulation may complete without overflowing.

        >>> diamond = sm.CycleChain(rates='uniform')
        >>> diamond.base_flow
        array([0.5, 0.5, 0. , 0.5, 0.5])

        >>> diamond.run('longest_queue', seed=42, number_events=20000)
        True
        >>> diamond.simulation
        array([0.501 , 0.4914, 0.0018, 0.478 , 0.5014])

        A working candy. While candies are not good for greedy policies, the virtual queue is
        designed to deal with it.

        >>> candy = sm.HyperPaddle(rates=[1, 1, 1.1, 1, 1.1, 1, 1])
        >>> candy.base_flow
        array([0.95, 0.05, 0.05, 0.05, 0.05, 0.95, 1.  ])

        The above states that the target flow for the hyperedge of the candy (last entry) is 1.

        >>> candy.run('longest_queue', seed=42, number_events=20000)
        False
        >>> candy.simulator.logs['steps_done']
        10459
        >>> candy.simulation  # doctest: +NORMALIZE_WHITESPACE
        array([0.64227938, 0.37586767, 0.38757051, 0.40753418, 0.40891099,
           0.59202601, 0.2939478 ])

        A greedy simulator performs poorly on the hyperedge.

        >>> candy.run('virtual_queue', seed=42, number_events=20000)
        True
        >>> candy.simulation  # doctest: +NORMALIZE_WHITESPACE
        array([0.96048, 0.04104, 0.04428, 0.06084, 0.06084, 0.94464, 0.9846 ])

        The virtual queue simulator manages to cope with the target flow on the hyperedge.
        """
        simulator = class_converter(simulator, Simulator)
        self.simulator = simulator(self, **kwargs)
        self.simulator.run()
        self.simulation = self.simulator.compute_flow()
        self.base_flow = self.simulation
        return self.simulator.generator['number_events'] == self.simulator.logs['steps_done']
