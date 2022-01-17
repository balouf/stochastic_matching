import numpy as np
from scipy.optimize import linprog

from stochastic_matching.graphs.classes import neighbors, SimpleGraph

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

    >>> mat = np.array([[1e-12, -.3], [.8, -1e-13]])
    >>> clean_zeros(mat)
    >>> mat # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. , -0.3],
           [ 0.8,  0. ]])
    """
    matrix[abs(matrix[:]) < tol] = 0


status_names = {(False, False): "Nonjective",
                (True, False): "Injective-only",
                (False, True): "Surjective-only",
                (True, True): "Bijective"}


status_names_simple_connected = {(False, False): "Bipartite with cycle(s)",
                (True, False): "Tree",
                (False, True): "Non-bipartite polycyclic",
                (True, True): "Non-bipartite monocyclic"}


def incidence_analysis(incidence, tol=1e-10):
    """
    Performs linear algebra analysis on the incidence matrix.

    Parameters
    ----------
    incidence: :class:`~scipy.sparse.csr_matrix`
        Incidence matrix of the (hyper)graph.
    tol: :class:`float`
        Values of absolute value lower than `tol` are set to 0.

    Returns
    -------
    inv: :class:`~numpy.ndarray`
        Pseudo-inverse of the incidence matrix.
    kernel: :class:`~numpy.ndarray`
        Kernel (right) of the incidence matrix. Determines injectivity.
    left_kernel: :class:`~numpy.ndarray`
        Left kernel of the incidence matrix. Determines surjectivity.
    status: :class:`tuple` of :class:`bool`
        Tells the injectivivity and surjectivity of the graph.

    Examples
    --------

    Consider the fully inversible incidence of the paw graph (bijective graph, i.e. n=m, non bipartite).

    >>> from stochastic_matching.graphs.generators import tadpole_graph
    >>> paw = tadpole_graph()
    >>> inv, right, left, status  = incidence_analysis(paw.incidence)

    The inverse is:

    >>> inv
    array([[ 0.5,  0.5, -0.5,  0.5],
           [ 0.5, -0.5,  0.5, -0.5],
           [-0.5,  0.5,  0.5, -0.5],
           [ 0. ,  0. ,  0. ,  1. ]])

    We can check that it is indeed the inverse.

    >>> i = paw.incidence @ inv
    >>> clean_zeros(i)
    >>> i
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    Right kernel is trivial:

    >>> right.shape[0]
    0

    Left kernel is trivial:

    >>> left.shape[1]
    0

    Graph is bijective:

    >>> status_names[status]
    'Bijective'

    As the graph is simple and connected, there a more accurate description of the status:

    >>> status_names_simple_connected[status]
    'Non-bipartite monocyclic'

    Now consider a bipartite version, the banner graph :

    >>> banner = tadpole_graph(m=4)
    >>> inv, right, left, status  = incidence_analysis(banner.incidence)

    The pseudo-inverse is:

    >>> inv
    array([[ 0.35,  0.4 , -0.15, -0.1 ,  0.1 ],
           [ 0.45, -0.2 , -0.05,  0.3 , -0.3 ],
           [-0.15,  0.4 ,  0.35, -0.1 ,  0.1 ],
           [-0.05, -0.2 ,  0.45,  0.3 , -0.3 ],
           [-0.2 ,  0.2 , -0.2 ,  0.2 ,  0.8 ]])

    We can check that it is indeed not exactly the inverse.

    >>> i = banner.incidence @ inv
    >>> clean_zeros(i)
    >>> i
    array([[ 0.8,  0.2, -0.2,  0.2, -0.2],
           [ 0.2,  0.8,  0.2, -0.2,  0.2],
           [-0.2,  0.2,  0.8,  0.2, -0.2],
           [ 0.2, -0.2,  0.2,  0.8,  0.2],
           [-0.2,  0.2, -0.2,  0.2,  0.8]])

    Right kernel is not trivial because of the even cycle:

    >>> right.shape[0]
    1
    >>> right # doctest: +SKIP
    array([[ 0.5, -0.5, -0.5,  0.5,  0. ]])

    Left kernel is not trivial because of the bipartite degenerescence:

    >>> left.shape[1]
    1
    >>> left
    array([[ 0.4472136],
           [-0.4472136],
           [ 0.4472136],
           [-0.4472136],
           [ 0.4472136]])

    Status is nonjective (not injective nor bijective):

    >>> status_names[status]
    'Nonjective'

    As the graph is simple and connected, there a more accurate description of the status:

    >>> status_names_simple_connected[status]
    'Bipartite with cycle(s)'

    Consider now the diamond graph, surjective (n<m, non bipartite).

    >>> from stochastic_matching.graphs.generators import chained_cycle_graph
    >>> diamond = chained_cycle_graph()
    >>> inv, right, left, status = incidence_analysis(diamond.incidence)

    The inverse is:

    >>> inv
    array([[ 0.5 ,  0.25, -0.25,  0.  ],
           [ 0.5 , -0.25,  0.25,  0.  ],
           [-0.5 ,  0.5 ,  0.5 , -0.5 ],
           [ 0.  ,  0.25, -0.25,  0.5 ],
           [ 0.  , -0.25,  0.25,  0.5 ]])

    We can check that it is indeed the inverse.

    >>> i = diamond.incidence @ inv
    >>> clean_zeros(i)
    >>> i
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    There is a right kernel:

    >>> right.shape[0]
    1
    >>> right
    array([[ 0.5, -0.5,  0. , -0.5,  0.5]])


    The left kernel is trivial:

    >>> left.shape[1]
    0

    The diamond is surjective-only:

    >>> status_names[status]
    'Surjective-only'

    As the graph is simple and connected, there a more accurate description of the status:

    >>> status_names_simple_connected[status]
    'Non-bipartite polycyclic'

    Consider now a star graph, injective (tree).

    >>> from stochastic_matching.graphs.generators import star_graph
    >>> star = star_graph()
    >>> inv, right, left, status = incidence_analysis(star.incidence)

    The inverse is:

    >>> inv
    array([[ 0.25,  0.75, -0.25, -0.25],
           [ 0.25, -0.25,  0.75, -0.25],
           [ 0.25, -0.25, -0.25,  0.75]])

    We can check that it is indeed the **left** inverse.

    >>> i = inv @ star.incidence
    >>> clean_zeros(i)
    >>> i
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    The right kernel is trivial:

    >>> right.shape[0]
    0

    The left kernel shows the bibartite behavior:

    >>> left.shape[1]
    1
    >>> left
    array([[-0.5],
           [ 0.5],
           [ 0.5],
           [ 0.5]])

    The star is injective-only:

    >>> status_names[status]
    'Injective-only'

    As the graph is simple and connected, there a more accurate description of the status:

    >>> status_names_simple_connected[status]
    'Tree'

    Next, a surjective hypergraph:

    >>> from stochastic_matching.graphs.generators import fan
    >>> clover = fan()
    >>> inv, right, left, status = incidence_analysis(clover.incidence)

    Incidence matrix dimensions:

    >>> clover.incidence.shape
    (9, 10)

    The inverse dimensions:

    >>> inv.shape
    (10, 9)

    We can check that it is exactly the inverse, because there was no dimensionnality loss.

    >>> i = clover.incidence @ inv
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

    >>> right.shape[0]
    1

    Left kernel is trivial.

    >>> left.shape[1]
    0

    Status:

    >>> status_names[status]
    'Surjective-only'

    Lastly, observe a *bipartite* hypergraph (in the sense of with non-trivial left kernel).

    >>> clover = fan(cycle_size=4)
    >>> inv, right, left, status = incidence_analysis(clover.incidence)

    Incidence matrix dimensions:

    >>> clover.incidence.shape
    (12, 13)

    The inverse dimensions:

    >>> inv.shape
    (13, 12)

    We can check that it is not exactly the inverse.

    >>> (clover.incidence @ inv)[:4, :4]
    array([[ 0.83333333,  0.16666667, -0.16666667,  0.16666667],
           [ 0.16666667,  0.83333333,  0.16666667, -0.16666667],
           [-0.16666667,  0.16666667,  0.83333333,  0.16666667],
           [ 0.16666667, -0.16666667,  0.16666667,  0.83333333]])

    Right kernel is 3 dimensional:

    >>> right.shape[0]
    3

    Left kernel is 2-dimensional (this is a change compared to simple graph,
    where the left kernel dimension of a connected component is at most 1).

    >>> left.shape[1]
    2

    Status:

    >>> status_names[status]
    'Nonjective'
    """
    n, m = incidence.shape
    min_d = min(n, m)
    u, s, v = np.linalg.svd(incidence.toarray())
    clean_zeros(s, tol=tol)
    dia = np.zeros((m, n))
    dia[:min_d, :min_d] = np.diag([pseudo_inverse_scalar(e) for e in s])
    ev = np.zeros(m)
    ev[:len(s)] = s
    kernel = v[ev == 0, :]
    eu = np.zeros(n)
    eu[:len(s)] = s
    left_kernel = u[:, eu==0]
    pseud_inv = v.T @ dia @ u.T
    clean_zeros(pseud_inv)
    clean_zeros(kernel)
    injective = kernel.shape[0] == 0
    surjective = left_kernel.shape[1] == 0
    return pseud_inv, kernel, left_kernel, (injective, surjective)


def traversal(graph):
    """
    Using graph traversal, splits the graph into its connected components as a list of sets of nodes and edges.
    If the graph is simple, additional information obtained by the traversal are provided.

    Parameters
    ----------
    graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graphs.classes.HyperGraph`
        Graph to decompose.

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

    >>> import stochastic_matching as sm
    >>> sample = sm.concatenate([sm.cycle_graph(4), sm.complete_graph(4), sm.chained_cycle_graph(),
    ...          sm.tadpole_graph(), sm.star_graph()], 0 )
    >>> traversal(sample) # doctest: +NORMALIZE_WHITESPACE
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

    >>> pyramid = sm.concatenate([sm.cycle_graph(), sm.cycle_graph(5), sm.cycle_graph(5), sm.cycle_graph()], 2)
    >>> traversal(pyramid) # doctest: +NORMALIZE_WHITESPACE
    [{'nodes': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    'edges': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    'spanner': {0, 1, 3, 4, 5, 7, 8, 9, 11},
    'pivot': 2, 'seeds': {10, 12, 6},
    'type': 'Non-bipartite polycyclic'}]

    If the graph is treated as hypergraph, a lot less information is available.

    >>> traversal(sample.to_hypergraph()) # doctest: +NORMALIZE_WHITESPACE
    [{'nodes': {0, 1, 2, 3}, 'edges': {0, 1, 2, 3}},
    {'nodes': {4, 5, 6, 7}, 'edges': {4, 5, 6, 7, 8, 9}},
    {'nodes': {8, 9, 10, 11}, 'edges': {10, 11, 12, 13, 14}},
    {'nodes': {12, 13, 14, 15}, 'edges': {16, 17, 18, 15}},
    {'nodes': {16, 17, 18, 19}, 'edges': {19, 20, 21}}]
    """
    simple = type(graph) == SimpleGraph
    n, m = graph.incidence.shape
    unknown_nodes = {i for i in range(n)}
    if simple:
        spin = np.ones(n, dtype=bool) # Simple
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
            edges = neighbors(i, graph.incidence)
            for edge in edges:
                if edge in current_edges:
                    continue
                for j in neighbors(edge, graph.co_incidence):
                    if j in unknown_nodes:
                        buffer.add(j)
                        unknown_nodes.remove(j)
                        if simple:
                            spin[j] = not spin[i] # Simple
                            current_spanner.add(edge)
                current_edges.add(edge)
        cc = {'nodes': current_nodes, 'edges': current_edges}
        if simple:
            cc['spanner'] = current_spanner # Simple
            free_edges = current_edges - current_spanner
            for edge in free_edges:
                pair = neighbors(edge, graph.co_incidence)
                if spin[pair[0]] == spin[pair[1]]:
                    cc['pivot'] = edge
                    free_edges.discard(edge)
                    break
            else:
                cc['pivot'] = False
            cc['seeds'] = free_edges
            injective = len(free_edges)==0
            surjective = cc['pivot'] is not False
            cc['type'] = status_names_simple_connected[(injective, surjective)]
        res.append(cc)
    return res


def simple_right_kernel(right, seeds):
    """
    Parameters
    ----------
    right: :class:`~numpy.ndarray`
        Right kernel (i.e. edges kernel) of a simple graph.
    seeds: :class:`list` of :class:`int`
        Seed edges of the kernel space. Valid seeds can be obtained from
        :meth:`~stochastic_matching.analysis.connected_components`.

    Returns
    -------
    :class:`~numpy.ndarray`
        The kernel expressed as elements from the cycle space (even cycles and kayak paddles).

    Examples
    --------

    Start with the co-domino.

    >>> import stochastic_matching as sm
    >>> codomino = sm.concatenate([sm.cycle_graph(), sm.cycle_graph(4), sm.cycle_graph()], 2)
    >>> _, right, _, _ = incidence_analysis(codomino.incidence)

    A first possible decomposition with a square and an hex.

    >>> simple_right_kernel(right, [5, 7])
    array([[ 0,  0,  1, -1, -1,  1,  0,  0],
           [ 1, -1,  0, -1,  1,  0, -1,  1]])

    A second possible decomposition with a kayak paddle and a square.

    >>> simple_right_kernel(right, [0, 4])
    array([[ 1, -1,  1, -2,  0,  1, -1,  1],
           [ 0,  0, -1,  1,  1, -1,  0,  0]])

    Another example with the pyramid.

    >>> pyramid = sm.concatenate([sm.cycle_graph(), sm.cycle_graph(5),
    ...                           sm.cycle_graph(5), sm.cycle_graph()], 2)
    >>> _, right, _, _ = incidence_analysis(pyramid.incidence)

    A first decomposition: cycles of length 6, 8, and 10.

    >>> simple_right_kernel(right, [6, 10, 12])
    array([[ 1, -1,  0, -1,  1, -1,  1,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  1, -1, -1,  1,  0, -1,  1, -1,  1,  0,  0],
           [-1,  1,  0,  1, -1,  1,  0, -1, -1,  1,  0, -1,  1]])

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
    return  np.round(np.linalg.inv(right[:, seeds]) @ right).astype(int)


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

    >>> kernel = np.array([[ 0,  0,  1, -1, -1,  1,  0,  0],
    ...       [ 1, -1,  0, -1,  1,  0, -1,  1]])
    >>> kinv = kernel_inverse(kernel)
    >>> kernel @ (np.array([-1, 3]) @ kernel) @ kinv
    array([-1.,  3.])

    If the kernel basis is not orthogonal, it returns somethings more complex.

    >>> kernel = np.array([[ 1, -1,  1, -2,  0,  1, -1,  1],
    ...    [ 0,  0, -1,  1,  1, -1,  0,  0]])
    >>> kinv = kernel_inverse(kernel)
    >>> kernel @ (np.array([2, -1]) @ kernel) @ kinv
    array([ 2., -1.])
    """
    return np.linalg.inv(np.inner(kernel, kernel))


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

    >>> import stochastic_matching as sm
    >>> sample = sm.concatenate([sm.cycle_graph(4), sm.star_graph(5)], 0)
    >>> _, _, left, _ = incidence_analysis(sample.incidence)
    >>> left
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

    >>> simple_left_kernel(left)
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


def uniform_rate(graph):
    """
    Parameters
    ----------
    graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graphs.classes.HyperGraph`
        Graph for which rates are to be provided.

    Returns
    -------
    :class:`~numpy.ndarray`
        Uniform arrival rates

    Examples
    --------

    >>> from stochastic_matching import chained_cycle_graph
    >>> diamond = chained_cycle_graph()
    >>> uniform_rate(diamond)
    array([1., 1., 1., 1.])
    """
    return np.ones(graph.n)


def proportional_rates(graph):
    """
    Parameters
    ----------
    graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graphs.classes.HyperGraph`
        Graph for which rates are to be provided.

    Returns
    -------
    :class:`~numpy.ndarray`
        Degree-proportional arrival rates

    Examples
    --------

    >>> from stochastic_matching import chained_cycle_graph
    >>> diamond = chained_cycle_graph()
    >>> proportional_rates(diamond)
    array([2., 3., 3., 2.])
    """
    return graph.incidence @ np.ones(graph.m)


class Analyzer:
    """
    The analyzer class is the frontend for all theoretical properties of a matching problem.

    Parameters
    ----------
    graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graphs.classes.HyperGraph`, optional
        Graph to analyze.
    mu: :class:`~numpy.ndarray` or :class:`list` or :class:`str`, optional
        Arrival rates. You can use a specific rate vector or list.
        You can use `uniform` or `proportional` for uniform or degree-proportional allocation.
        Default to `proportional`, which makes the problem stabilizable if the graph is bijective.
    tol: :class:`float`, optional
        Values of absolute value lower than `tol` are set to 0.

    Examples
    --------

    The following examples are about stability:

    Is a triangle that checks triangular inequality stable?

    >>> from stochastic_matching import cycle_graph
    >>> problem = Analyzer(cycle_graph(), mu='uniform')
    >>> problem.is_stable
    True

    >>> problem.graph_type
    'Bijective'

    We can look at the base flow (based on Moore-Penrose inverse).

    >>> problem.base_flow
    array([0.5, 0.5, 0.5])

    As the graph is bijective, all optimizations will yield the same flow.

    >>> problem.incompressible_flow()
    array([0.5, 0.5, 0.5])

    >>> problem.optimize_edge(0)
    array([0.5, 0.5, 0.5])

    What if the triangular inequality does not hold?

    >>> problem.fit(mu=[1, 3, 1])
    >>> problem.is_stable
    False

    We can look at the base flow (based on Moore-Penrose inverse).

    >>> problem.base_flow
    array([ 1.5, -0.5,  1.5])

    Now a bipartite example.

    >>> from stochastic_matching import tadpole_graph
    >>> problem.fit(graph=tadpole_graph(m=4))
    >>> problem.fit(mu='proportional')

    Notice that we have a perfectly working solution with respect to conservation law.

    >>> problem.base_flow
    array([1., 1., 1., 1., 1.])

    However, the left kernel is not trivial.

    >>> problem.left_kernel
    array([[ 1],
           [-1],
           [ 1],
           [-1],
           [ 1]])

    As a consequence, stability is False.

    >>> problem.is_stable
    False

    >>> problem.graph_type
    'Nonjective'

    Note that the base flow can be negative even if there is a positive solution.

    >>> from stochastic_matching import chained_cycle_graph
    >>> diamond = chained_cycle_graph()
    >>> problem = Analyzer(diamond, [5, 6, 2, 1])
    >>> problem.base_flow
    array([ 3.5,  1.5,  1. ,  1.5, -0.5])
    >>> problem.is_stable
    True
    >>> np.round(problem.maximin_flow()*10)/10
    array([4.5, 0.5, 1. , 0.5, 0.5])

    >>> problem.incompressible_flow()
    array([4., 0., 1., 0., 0.])

    >>> problem.graph_type
    'Surjective-only'
    """

    def __init__(self, graph=None, mu=None, tol=1e-10):
        self.tol = tol
        self.simple = None
        self.m = None
        self.n = None
        self.degree = None
        self.left_kernel = None
        self.right_kernel = None
        self._right_inverse = None
        self.pseudo_inverse = None
        self.status = None
        self.connected_components = None
        self.base_flow = None
        self.positive_solution_exists = None
        self.fit(graph=graph, mu=mu)

    @property
    def is_stable(self):
        """
        :class:`bool`: Tells whether a stable policy be enforced for the graph and arrival rate.
        """
        self.maximin_flow()
        return self.status[1] and self.positive_solution_exists

    @property
    def graph_type(self):
        """
        :class:`str`: Injectivity/surjectivity of the graph.
        """
        return status_names[self.status]

    def fit(self, graph=None, mu=None):
        """
        Compute internal attributes (pseudo-inverse, kernel, base solution) for graph and/or rate.
        If `graph` is provided without `mu`, uniform rate is assumed.

        Parameters
        ----------
        graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graphs.classes.HyperGraph`, optional
            Graph to analyze.
        mu: :class:`~numpy.ndarray` or :class:`list` or :class:`str`, optional
            Arrival rates. You can use a specific rate vector or list.
            You can use `uniform` or `proportional` for uniform or degree-proportional allocation.
            Default to `proportional`, which makes the problem stabilizable if the graph is bijective.

        Returns
        -------
        None
        """
        if graph is not None:
            # self.graph = graph
            self.simple = type(graph) == SimpleGraph
            self.pseudo_inverse, self.right_kernel, \
            self.left_kernel, self.status = incidence_analysis(graph.incidence, tol=self.tol)
            self.n = graph.n
            self.m = graph.m
            self.degree = graph.incidence @ np.ones(graph.m)
            self.connected_components = traversal(graph)
            if self.simple:
                seeds = [i for c in self.connected_components for i in c['seeds']]
                self.right_kernel = simple_right_kernel(self.right_kernel, seeds)
                self.left_kernel = simple_left_kernel(self.left_kernel)
            self._right_inverse = kernel_inverse(self.right_kernel)
            if mu is None:
                mu = self.degree
        if mu is not None:
            if isinstance(mu, str):
                if mu == 'uniform':
                    mu = np.ones(self.n)
                else:
                    mu = self.degree
            self.base_flow = self.pseudo_inverse @ mu
            clean_zeros(self.base_flow, tol=self.tol)

    def change_kernel_basis(self, seeds):
        """
        Change the cycle space using provided seeds to span the basis.

        Parameters
        ----------
        seeds: :class:`list` of :class:`int`
            Seed edges (cf https://hal.archives-ouvertes.fr/hal-03502084).

        Returns
        -------
        None

        Examples
        --------

        >>> import stochastic_matching as sm
        >>> codomino = sm.concatenate([sm.cycle_graph(), sm.cycle_graph(4), sm.cycle_graph()], 2)
        >>> problem = Analyzer(codomino)
        >>> problem.right_kernel # doctest: +NORMALIZE_WHITESPACE
        array([[ 0,  0,  1, -1, -1,  1,  0,  0],
               [ 1, -1,  0, -1,  1,  0, -1,  1]])
        >>> problem.change_kernel_basis([0, 4])
        >>> problem.right_kernel # doctest: +NORMALIZE_WHITESPACE
        array([[ 1, -1,  1, -2,  0,  1, -1,  1],
               [ 0,  0, -1,  1,  1, -1,  0,  0]])

        Be careful to choose valid seeds (e.g. the graph with the seeds removed must be injective).

        >>> problem.change_kernel_basis([0, 1])
        Traceback (most recent call last):
        ...
        numpy.linalg.LinAlgError: Singular matrix

        Changing basis is only available for simple graphs.

        >>> problem.fit(codomino.to_hypergraph())
        >>> problem.change_kernel_basis([0, 4])
        Traceback (most recent call last):
        ...
        NotImplementedError
        """
        if not self.simple:
            raise NotImplementedError
        self.right_kernel = simple_right_kernel(self.right_kernel, seeds)
        self._right_inverse = kernel_inverse(self.right_kernel)

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

        >>> import stochastic_matching as sm
        >>> codomino = sm.concatenate([sm.cycle_graph(), sm.cycle_graph(4), sm.cycle_graph()], 2)
        >>> problem = Analyzer(codomino, [3, 12, 3, 3, 12, 3])
        >>> problem.change_kernel_basis([0, 4])
        >>> problem.right_kernel # doctest: +NORMALIZE_WHITESPACE
        array([[ 1, -1,  1, -2,  0,  1, -1,  1],
           [ 0,  0, -1,  1,  1, -1,  0,  0]])

        Consider the base flow (Moore-Penrose inverse) and the maximin flow.

        >>> moore = problem.base_flow
        >>> moore
        array([3., 0., 3., 6., 0., 3., 0., 3.])

        >>> maxmin = problem.maximin_flow()
        >>> maxmin
        array([2., 1., 1., 9., 1., 1., 1., 2.])

        As the Moore-Penrose inverse is the base flow, its coordinates are obviously null.

        >>> problem.edge_to_kernel(moore)
        array([0., 0.])

        As for maximin, one can check that the following kernel coordinates transform Moore-Penrose into it:

        >>> problem.edge_to_kernel(maxmin)
        array([-1.,  1.])

        If we change the base flow to maximin, we will see the coordinates shifted by (1, -1):

        >>> problem.base_flow = maxmin
        >>> problem.edge_to_kernel(moore)
        array([ 1., -1.])
        >>> problem.edge_to_kernel(maxmin)
        array([0., 0.])
        """
        res = (self.right_kernel @ (edge - self.base_flow)) @ self._right_inverse
        clean_zeros(res, tol=self.tol)
        return res

    def kernel_to_edge(self, kernel):
        """
        Parameters
        ----------
        kernel: :class:`~numpy.ndarray`
            A flow vector in kernel coordinates.

        Returns
        -------
        :class:`~numpy.ndarray`
            The same flow vector in edge coordinates, based on the current base flow and right kernel.

        Examples
        --------

        Consider the codomino graph with a kernel with a kayak paddle.

        >>> import stochastic_matching as sm
        >>> codomino = sm.concatenate([sm.cycle_graph(), sm.cycle_graph(4), sm.cycle_graph()], 2)
        >>> problem = Analyzer(codomino, [3, 12, 3, 3, 12, 3])
        >>> problem.right_kernel # doctest: +NORMALIZE_WHITESPACE
        array([[ 0,  0,  1, -1, -1,  1,  0,  0],
           [ 1, -1,  0, -1,  1,  0, -1,  1]])

        Consider the base flow (Moore-Penrose inverse) and the maximin flow.

        >>> moore = problem.base_flow
        >>> moore
        array([3., 0., 3., 6., 0., 3., 0., 3.])

        >>> maxmin = problem.maximin_flow()
        >>> maxmin
        array([2., 1., 1., 9., 1., 1., 1., 2.])

        As the Moore-Penrose inverse is the base flow, it is (0, 0) in kernel coordinates.

        >>> problem.kernel_to_edge([0, 0])
        array([3., 0., 3., 6., 0., 3., 0., 3.])

        As for maximin, (-2, -1) seems to be its kernel coordinates.

        >>> problem.kernel_to_edge([-2, -1])
        array([2., 1., 1., 9., 1., 1., 1., 2.])

        If we change the kernel space, the kernel coordinates change as well.

        >>> problem.change_kernel_basis([0, 4])
        >>> problem.right_kernel # doctest: +NORMALIZE_WHITESPACE
        array([[ 1, -1,  1, -2,  0,  1, -1,  1],
           [ 0,  0, -1,  1,  1, -1,  0,  0]])
        >>> problem.kernel_to_edge([0, 0])
        array([3., 0., 3., 6., 0., 3., 0., 3.])
        >>> problem.kernel_to_edge([-1, 1])
        array([2., 1., 1., 9., 1., 1., 1., 2.])
        """
        res = (kernel @ self.right_kernel) + self.base_flow
        clean_zeros(res, tol=self.tol)
        return res

    def kernel_dict(self, flow=None):
        """
        Parameters
        ----------
        flow: :class:`~numpy.ndarray` ot :class:`bool`, optional
            Base flow of the kernel representation. If False, no base flow is displayed, only the kernel shifts.
            If no flow is given, the current base flow is used.

        Returns
        -------
        :class:`list` of :class:`dict`
            An edge description dictionary to pass to :meth:`~stochastic_matching.graphs.classes.GenericGraph.show`.

        Examples
        --------

        >>> import stochastic_matching as sm
        >>> diamond = sm.chained_cycle_graph()
        >>> problem = Analyzer(diamond)
        >>> problem.base_flow
        array([1., 1., 1., 1., 1.])
        >>> problem.kernel_dict()
        [{'label': '1+α1'}, {'label': '1-α1'}, {'label': '1', 'color': 'black'}, {'label': '1-α1'}, {'label': '1+α1'}]
        >>> problem.kernel_dict(flow=False)
        [{'label': '+α1'}, {'label': '-α1'}, {'label': '', 'color': 'black'}, {'label': '-α1'}, {'label': '+α1'}]
        >>> min_flow = problem.optimize_edge(0, -1)
        >>> min_flow
        array([0., 2., 1., 2., 0.])
        >>> problem.kernel_dict(flow=min_flow)
        [{'label': '+α1'}, {'label': '2-α1'}, {'label': '1', 'color': 'black'}, {'label': '2-α1'}, {'label': '+α1'}]

        >>> kayak = sm.kayak_paddle_graph(l=3)
        >>> problem.fit(kayak)
        >>> problem.kernel_dict() # doctest: +NORMALIZE_WHITESPACE
        [{'label': '1-α1'}, {'label': '1+α1'}, {'label': '1+α1'},
         {'label': '1-2α1'}, {'label': '1+2α1'}, {'label': '1-2α1'},
         {'label': '1+α1'}, {'label': '1+α1'}, {'label': '1-α1'}]
        """
        d, m = self.right_kernel.shape
        edge_description = [dict() for _ in range(m)]
        for e in range(m):
            label = ""
            for i in range(d):
                alpha = self.right_kernel[i, e]
                if alpha == 0:
                    continue
                if alpha == 1:
                    label += f"+"
                elif alpha == -1:
                    label += f"-"
                else:
                    label += f"{alpha:+.3g}"
                label += f"α{i + 1}"
            edge_description[e]['label'] = label
            if not label:
                edge_description[e]['color'] = 'black'
        if flow is False:
            return edge_description
        if flow is None:
            flow = self.base_flow
        for e, dico in enumerate(edge_description):
            if np.abs(flow[e]) > self.tol:
                dico['label'] = f"{flow[e]:.3g}{dico['label']}"
        return edge_description

    def show_solutions(self, graph, mu=None):
        """
        Parameters
        ----------
        graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graphs.classes.HyperGraph`, optional
            Graph to use.
        mu: :class:`~numpy.ndarray`, optional
            Arrival rates

        Returns
        -------
        :class:`~IPython.display.HTML`

        Examples
        --------

        >>> import stochastic_matching as sm
        >>> diamond = sm.chained_cycle_graph()
        >>> problem = Analyzer(diamond)
        >>> problem.show_solutions(diamond)
        <IPython.core.display.HTML object>
        >>> problem.show_solutions(diamond, problem.degree)
        <IPython.core.display.HTML object>
        """
        if mu is not None:
            nodes_description = [{'label': f"{mu[i]:.3g}"} for i in range(graph.n)]
        else:
            nodes_description = None
        graph.show(nodes_dict=nodes_description, edges_dict=self.kernel_dict())


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
        d, m = self.right_kernel.shape
        if d == 0:
            return self.base_flow
        else:
            optimizer = linprog(c=-sign * self.right_kernel[:, edge],
                                A_ub=-self.right_kernel.T,
                                b_ub=self.base_flow,
                                bounds=[(None, None)] * d
                                )
            clean_zeros(optimizer.slack, tol=self.tol)
            return optimizer.slack

    def maximin_flow(self):
        """
        Maximizes the minimal flow over all edges and records whether a positive solution was found.

        Returns
        -------
        :class:`~numpy.ndarray`
            Optimized flow.
        """
        d, m = self.right_kernel.shape
        if d == 0:
            # noinspection PyUnresolvedReferences
            self.positive_solution_exists = np.amin(self.base_flow) > 0
            return self.base_flow
        else:
            c = np.zeros(d + 1)
            c[d] = 1
            a_ub = -np.vstack([self.right_kernel, np.ones(m)]).T
            optimizer = linprog(c=c,
                                A_ub=a_ub,
                                b_ub=self.base_flow,
                                bounds=[(None, None)] * (d + 1)
                                )
            flow = optimizer.slack - optimizer.x[-1]
            clean_zeros(flow, tol=self.tol)
            self.positive_solution_exists = (optimizer.x[-1] < 0)
            return flow

    def incompressible_flow(self):
        """
        Finds the minimal flow that must pass through each edge. This is currently done in a *brute force* way
        by minimizing every edges.

        Returns
        -------
        :class:`~numpy.ndarray`
            Unavoidable flow.
        """
        d, m = self.right_kernel.shape
        if d == 0:
            return self.base_flow
        else:
            flow = np.zeros(m)
            for edge in range(m):
                flow[edge] = self.optimize_edge(edge, -1)[edge]
            clean_zeros(flow, tol=self.tol)
            return flow



def inverse_incidence(incidence, tol=1e-10):
    """
    *Reverse* the incidence matrix.

    Parameters
    ----------
    incidence: :class:`~scipy.sparse.csr_matrix`
        Incidence matrix of the (hyper)graph.
    tol: :class:`float`
        Values of absolute value lower than `tol` are set to 0.

    Returns
    -------
    inv: :class:`~numpy.ndarray`
        Pseudo-inverse of the incidence matrix.
    kernel: :class:`~numpy.ndarray`
        Pseudo-kernel of the incidence matrix.
    bipartite: :class:`bool`
        Tells whether the dimension of the kernel is greater than (m-n),
        which hints at bipartite structures for simple graphs.

    Examples
    --------

    Consider a fully inversible incidence (n=m, non bipartite).

    >>> from stochastic_matching.graphs.generators import tadpole_graph
    >>> p = tadpole_graph()
    >>> inv, k, b  =inverse_incidence(p.incidence)

    The inverse is:

    >>> inv
    array([[ 0.5,  0.5, -0.5,  0.5],
           [ 0.5, -0.5,  0.5, -0.5],
           [-0.5,  0.5,  0.5, -0.5],
           [ 0. ,  0. ,  0. ,  1. ]])

    We can check that it is indeed the inverse.

    >>> i = p.incidence.dot(inv)
    >>> clean_zeros(i)
    >>> i
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    Kernel is trivial:

    >>> k
    array([], shape=(0, 4), dtype=float64)

    No bipartite behavior:

    >>> b
    False

    Now consider a bipartite version :

    >>> p = tadpole_graph(m=4)
    >>> inv, k, b  = inverse_incidence(p.incidence)

    The pseudo-inverse is:

    >>> inv
    array([[ 0.35,  0.4 , -0.15, -0.1 ,  0.1 ],
           [ 0.45, -0.2 , -0.05,  0.3 , -0.3 ],
           [-0.15,  0.4 ,  0.35, -0.1 ,  0.1 ],
           [-0.05, -0.2 ,  0.45,  0.3 , -0.3 ],
           [-0.2 ,  0.2 , -0.2 ,  0.2 ,  0.8 ]])

    We can check that it is indeed not exactly the inverse.

    >>> i = p.incidence.dot(inv)
    >>> clean_zeros(i)
    >>> i
    array([[ 0.8,  0.2, -0.2,  0.2, -0.2],
           [ 0.2,  0.8,  0.2, -0.2,  0.2],
           [-0.2,  0.2,  0.8,  0.2, -0.2],
           [ 0.2, -0.2,  0.2,  0.8,  0.2],
           [-0.2,  0.2, -0.2,  0.2,  0.8]])

    Kernel is not trivial because of the bipartite degenerescence:

    >>> k.shape
    (1, 5)

    >>> k # doctest: +SKIP
    array([[ 0.5, -0.5, -0.5,  0.5,  0. ]])

    Bipartite behavior:

    >>> b
    True

    Consider now the braess graph (n<m, non bipartite).

    >>> from stochastic_matching.graphs.generators import bicycle_graph
    >>> braess = bicycle_graph()
    >>> inv, k, b  =inverse_incidence(braess.incidence)

    The inverse is:

    >>> inv
    array([[ 0.5 ,  0.25, -0.25,  0.  ],
           [ 0.5 , -0.25,  0.25,  0.  ],
           [-0.5 ,  0.5 ,  0.5 , -0.5 ],
           [ 0.  ,  0.25, -0.25,  0.5 ],
           [ 0.  , -0.25,  0.25,  0.5 ]])

    We can check that it is indeed the inverse.

    >>> i = braess.incidence.dot(inv)
    >>> clean_zeros(i)
    >>> i
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

    There is a kernel:

    >>> k
    array([[ 0.5, -0.5,  0. , -0.5,  0.5]])

    No bipartite behavior:

    >>> b
    False

    Next, a well formed hypergraph:

    >>> from stochastic_matching.graphs.generators import fan
    >>> clover = fan()
    >>> inv, k, b  =inverse_incidence(clover.incidence)

    Incidence matrix dimensions:

    >>> clover.incidence.shape
    (9, 10)

    The inverse dimensions:

    >>> inv.shape
    (10, 9)

    We can check that it is exactly the inverse, because there was no dimensionnality loss.

    >>> i = clover.incidence.dot(inv)
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

    Kernel is 1 dimensional:

    >>> k.shape
    (1, 10)

    Non-bipartite behavior:

    >>> b
    False

    Lastly, observe a *bipartite* hypergraph.

    >>> clover = fan(cycle_size=4)
    >>> inv, k, b  =inverse_incidence(clover.incidence)

    Incidence matrix dimensions:

    >>> clover.incidence.shape
    (12, 13)

    The inverse dimensions:

    >>> inv.shape
    (13, 12)

    We can check that it is not exactly the inverse.

    >>> clover.incidence.dot(inv)[:4, :4]
    array([[ 0.83333333,  0.16666667, -0.16666667,  0.16666667],
           [ 0.16666667,  0.83333333,  0.16666667, -0.16666667],
           [-0.16666667,  0.16666667,  0.83333333,  0.16666667],
           [ 0.16666667, -0.16666667,  0.16666667,  0.83333333]])

    Kernel is 3 dimensional:

    >>> k.shape
    (3, 13)

    Bipartite behavior:

    >>> b
    True
    """
    n, m = incidence.shape
    min_d = min(n, m)
    u, s, v = np.linalg.svd(incidence.toarray())
    clean_zeros(s, tol=tol)
    dia = np.zeros((m, n))
    dia[:min_d, :min_d] = np.diag([pseudo_inverse_scalar(e) for e in s])
    ev = np.zeros(m)
    ev[:len(s)] = s
    kernel = v[ev == 0, :]
    bipartite = ((m - kernel.shape[0]) < n)
    pseud_inv = np.dot(v.T, np.dot(dia, u.T))
    clean_zeros(pseud_inv)
    clean_zeros(kernel)
    return pseud_inv, kernel, bipartite


class Spectral:
    """
    The spectral class handles all the flow computations based on the conservation law.

    Parameters
    ----------
    graph: :class:`~stochastic_matching.graphs.classes.GenericGraph`, optional
        Graph to analyze.
    mu: :class:`~numpy.ndarray` or :class:`list`, optional
        Arrival rates.
    tol: :class:`float`, optional
        Values of absolute value lower than `tol` are set to 0.

    Examples
    --------

    The following examples are about stability:

    Is a triangle that checks triangular inequality stable?

    >>> from stochastic_matching import tadpole_graph
    >>> spec = Spectral(tadpole_graph(n=0), [3, 4, 5])
    >>> spec.is_stable
    True

    What if the triangular inequality does not hold?

    >>> spec.fit(mu=[1, 2, 1])
    >>> spec.is_stable
    False

    Now a bipartite example.

    >>> spec.fit(graph=tadpole_graph(m=4))
    >>> spec.fit(mu=[1, 1, 1, 2, 1])

    Notice that we have a perfectly working solution with respect to conservation law.

    >>> spec.base_flow
    array([0.5, 0.5, 0.5, 0.5, 1. ])

    However, the kernel is degenerated.

    >>> spec.kernel.shape
    (1, 5)

    >>> spec.kernel # doctest: +SKIP
    array([[ 0.5, -0.5, -0.5,  0.5,  0. ]])

    As a consequence, stability is False.

    >>> spec.is_stable
    False
    """

    def __init__(self, graph=None, mu=None, tol=1e-10):
        self.tol = tol
        self.kernel = None
        self.pseudo_inverse = None
        self.base_flow = None
        self.bipartite = None
        self.positive_solution_exists = None
        self.fit(graph=graph, mu=mu)

    @property
    def is_stable(self):
        """
        :class:`bool`: Tells whether a stable policy be enforced for the graph and arrival rate.
        """
        self.maximin_flow()
        return (not self.bipartite) and self.positive_solution_exists

    def fit(self, graph=None, mu=None):
        """
        Compute internal attributes (pseudo-inverse, kernel, base solution) for graph and/or rate.
        If `graph` is provided without `mu`, uniform rate is assumed.

        Parameters
        ----------
        graph: :class:`~stochastic_matching.graphs.classes.GenericGraph`, optional
            Graph to analyze.
        mu: :class:`~numpy.ndarray` or :class:`list`, optional
            Arrival rates.

        Returns
        -------
        None
        """
        if graph is not None:
            self.pseudo_inverse, self.kernel, self.bipartite = inverse_incidence(graph.incidence, tol=self.tol)
            if mu is None:
                mu = np.ones(self.pseudo_inverse.shape[1])
        if mu is not None:
            self.base_flow = np.dot(self.pseudo_inverse, mu)
            clean_zeros(self.base_flow, tol=self.tol)

    def optimize_edge(self, edge, sign):
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
        d, m = self.kernel.shape
        if d == 0:
            return self.base_flow
        else:
            optimizer = linprog(c=-sign * self.kernel[:, edge],
                                A_ub=-self.kernel.T,
                                b_ub=self.base_flow,
                                bounds=[(None, None)] * self.kernel.shape[0]
                                )
            clean_zeros(optimizer.slack, tol=self.tol)
            return optimizer.slack

    def maximin_flow(self):
        """
        Maximizes the minimal flow over all edges and records whether a positive solution was found.

        Returns
        -------
        :class:`~numpy.ndarray`
            Optimized flow.
        """
        d, m = self.kernel.shape
        if d == 0:
            # noinspection PyUnresolvedReferences
            self.positive_solution_exists = np.amin(self.base_flow) > 0
            return self.base_flow
        else:
            c = np.zeros(d + 1)
            c[d] = 1
            a_ub = -np.vstack([self.kernel, np.ones(m)]).T
            optimizer = linprog(c=c,
                                A_ub=a_ub,
                                b_ub=self.base_flow,
                                bounds=[(None, None)] * (d + 1)
                                )
            flow = optimizer.slack - optimizer.x[-1]
            clean_zeros(flow, tol=self.tol)
            self.positive_solution_exists = (optimizer.x[-1] < 0)
            return flow

    def incompressible_flow(self):
        """
        Finds the minimal flow that must pass through each edge. This is currently done in a *brute force* way
        by minimizing every edges.

        Returns
        -------
        :class:`~numpy.ndarray`
            Unavoidable flow.
        """
        d, m = self.kernel.shape
        if d == 0:
            return self.base_flow
        else:
            flow = np.zeros(m)
            for edge in range(m):
                flow[edge] = self.optimize_edge(edge, -1)[edge]
            clean_zeros(flow, tol=self.tol)
            return flow
