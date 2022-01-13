import numpy as np
from scipy.optimize import linprog


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
