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


status_names_simple_connected = {(False, False): "Bipartite with cycle",
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
    'Bipartite with cycle'

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


def connected_components(graph):
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
    >>> connected_components(sample) # doctest: +NORMALIZE_WHITESPACE
    [{'nodes': {0, 1, 2, 3}, 'edges': {0, 1, 2, 3},
    'spanner': {0, 1, 2}, 'pivot': False, 'seeds': {3},
    'type': 'Bipartite with cycle'},
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
    >>> connected_components(pyramid) # doctest: +NORMALIZE_WHITESPACE
    [{'nodes': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    'edges': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    'spanner': {0, 1, 3, 4, 5, 7, 8, 9, 11},
    'pivot': 2, 'seeds': {10, 12, 6},
    'type': 'Non-bipartite polycyclic'}]

    If the graph is treated as hypergraph, a lot less information is available.

    >>> connected_components(sample.to_hypergraph()) # doctest: +NORMALIZE_WHITESPACE
    [{'nodes': {0, 1, 2, 3}, 'edges': {0, 1, 2, 3}},
    {'nodes': {4, 5, 6, 7}, 'edges': {4, 5, 6, 7, 8, 9}},
    {'nodes': {8, 9, 10, 11}, 'edges': {10, 11, 12, 13, 14}},
    {'nodes': {12, 13, 14, 15}, 'edges': {16, 17, 18, 15}},
    {'nodes': {16, 17, 18, 19}, 'edges': {19, 20, 21}}]
    """
    simple = type(graph) == SimpleGraph
    n, m = graph.incidence.shape
    unknown_nodes = {i for i in range(n)}
    unknown_edges = {j for j in range(m)}
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
                if edge in unknown_edges:
                    unknown_edges.add(edge)
                else:
                    break
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
    >>> problem = Analyzer(cycle_graph(), [3, 4, 5])
    >>> problem.is_stable
    True

    What if the triangular inequality does not hold?

    >>> problem.fit(mu=[1, 2, 1])
    >>> problem.is_stable
    False

    Now a bipartite example.

    >>> from stochastic_matching import tadpole_graph
    >>> problem.fit(graph=tadpole_graph(m=4))
    >>> problem.fit(mu=[1, 1, 1, 2, 1])

    Notice that we have a perfectly working solution with respect to conservation law.

    >>> problem.base_flow
    array([0.5, 0.5, 0.5, 0.5, 1. ])

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

    Note that the base flow can be negative even if there is a positive solution.

    >>> from stochastic_matching import chained_cycle_graph
    >>> diamond = chained_cycle_graph()
    >>> problem = Analyzer(diamond, [5, 5, 1, 1])
    >>> problem.base_flow
    array([ 3.5,  1.5,  0. ,  1.5, -0.5])
    >>> problem.is_stable
    False
    >>> problem.maximin_flow()
    array([4.34533263, 0.65466737, 0.        , 0.65466737, 0.34533263])
    >>> problem.fit(mu=[5, 6, 2, 1])
    >>> problem.base_flow
    array([ 3.5,  1.5,  1. ,  1.5, -0.5])
    >>> problem.is_stable
    True
    >>> np.round(problem.maximin_flow()*10)/10
    array([4.5, 0.5, 1. , 0.5, 0.5])
    """

    def __init__(self, graph=None, mu=None, tol=1e-10):
        self.tol = tol
        self.graph = None
        self.simple = None
        self.left_kernel = None
        self.right_kernel = None
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
            self.graph = graph
            self.simple = type(graph) == SimpleGraph
            self.pseudo_inverse, self.right_kernel, \
            self.left_kernel, self.status = incidence_analysis(graph.incidence, tol=self.tol)
            self.connected_components = connected_components(graph)
            if self.simple:
                seeds = [i for c in self.connected_components for i in c['seeds']]
                self.right_kernel = simple_right_kernel(self.right_kernel, seeds)
                self.left_kernel = simple_left_kernel(self.left_kernel)
            if mu is None:
                mu = proportional_rates(graph)
        if mu is not None:
            if isinstance(mu, str):
                if mu == 'uniform':
                    mu = uniform_rate(self.graph)
                else:
                    mu = proportional_rates(self.graph)
            self.base_flow = self.pseudo_inverse @ mu
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
