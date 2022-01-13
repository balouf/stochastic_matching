import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from stochastic_matching.graphs.display import HYPER_GRAPH_VIS_OPTIONS, vis_show


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

    Convert a diamond graph from adjacency to incidence.

    >>> from stochastic_matching.graphs.generators import bicycle_graph
    >>> diamond = bicycle_graph().adjacency
    >>> diamond
    array([[0, 1, 1, 0],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [0, 1, 1, 0]])
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

    >>> from stochastic_matching.graphs.generators import bicycle_graph
    >>> diamond = bicycle_graph().incidence.toarray().astype('int')
    >>> diamond
    array([[1, 1, 0, 0, 0],
           [1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1],
           [0, 0, 0, 1, 1]])
    >>> incidence_to_adjacency(diamond)
    array([[0, 1, 1, 0],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [0, 1, 1, 0]])

    An error occurs if one tries to convert a hypergraph.

    >>> from stochastic_matching.graphs.generators import hyper_paddle
    >>> candy = hyper_paddle().incidence.toarray().astype('int')
    >>> candy
    array([[1, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1, 0, 1],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 1, 0]])
    >>> incidence_to_adjacency(candy)
    Traceback (most recent call last):
    ...
    ValueError: The incidence matrix does not correspond to a simple graph.
    """
    # noinspection PyUnresolvedReferences
    if not np.all(np.sum(incidence, axis=0) == 2):
        raise ValueError("The incidence matrix does not correspond to a simple graph.")
    incidence = csc_matrix(incidence)
    n, m = incidence.shape
    adjacency = np.zeros((n, n), dtype=int)
    for j in range(m):
        e = neighbors(j, incidence)
        adjacency[e[0], e[1]] = 1
        adjacency[e[1], e[0]] = 1
    return adjacency


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


class GenericGraph:
    """
    Abstract class for :class:`~stochastic_matching.graphs.classes.SimpleGraph` and
    :class:`~stochastic_matching.graphs.classes.HyperGraph`.

    Parameters
    ----------
    incidence: :class:`~numpy.ndarray`
        Incidence matrix.

    Attributes
    ----------
    n: :class:`int`
        Number of nodes.
    m: :class:`int`
        Number of edges.
    incidence: :class:`~scipy.sparse.csr_matrix`
        :class:`~scipy.sparse.csr_matrix` view of the incidence matrix.
    co_incidence: :class:`~scipy.sparse.csc_matrix`
        :class:`~scipy.sparse.csc_matrix` view of the incidence matrix.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)
    """
    def __init__(self, incidence=None, names=None):
        self.n = None
        self.m = None
        self.names = names
        self.co_incidence = None
        self.incidence = incidence
        if names == "alpha":
            self.names = CharMaker()

    @property
    def incidence(self):
        return self.__incidence

    @incidence.setter
    def incidence(self, incidence):
        if incidence is None:
            self.__incidence = None
        else:
            self.n, self.m = incidence.shape
            self.__incidence = csr_matrix(incidence)
            self.co_incidence = csc_matrix(incidence)

    def int_2_str(self, i):
        if self.names is None:
            return str(i)
        else:
            return self.names[i]

    def vis_inputs(self, options, nodes_dict, edges_dict):
        raise NotImplementedError

    def show(self, options=None, nodes_dict=None, edges_dict=None):
        """
        Shows the simple graph.

        Parameters
        ----------
        options: :class:`dict`
            Additional / overriding options to pass to the vis engine.
        nodes_dict: :class:`list` of :class:`dict`
            Additional / overriding attributes for the nodes.
        edges_dict: :class:`list` of :class:`dict`
            Additional / overriding attributes for the edges.

        Returns
        -------
        :class:`~IPython.display.HTML`

        Examples
        ---------
        >>> from stochastic_matching.graphs.generators import tadpole_graph
        >>> tadpole_graph().show()
        <IPython.core.display.HTML object>
        """
        vis_nodes, vis_edges, vis_options = self.vis_inputs(options=options,
                                                            nodes_dict=nodes_dict,
                                                            edges_dict=edges_dict)
        vis_show(vis_nodes, vis_edges, vis_options)


class HyperGraph(GenericGraph):
    """
    Parameters
    ----------
    incidence: :class:`~numpy.ndarray`
        Incidence matrix

    Attributes
    ----------
    n: :class:`int`
        Number of nodes.
    m: :class:`int`
        Number of edges.
    incidence: :class:`~scipy.sparse.csr_matrix`
        :class:`~scipy.sparse.csr_matrix` view of the incidence matrix.
    co_incidence: :class:`~scipy.sparse.csc_matrix`
        :class:`~scipy.sparse.csc_matrix` view of the incidence matrix.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)
    """
    def __init__(self, incidence=None, names=None):
        super().__init__(incidence=incidence, names=names)

    def to_simplegraph(self):
        """
        Converts to simple graph. Raises an error if not possible.

        Returns
        -------
        :class:`~stochastic_matching.graphs.classes.SimpleGraph`

        Examples
        --------

        We will use the traditional diamond graph, represented as .

        >>> from stochastic_matching.graphs.generators import bicycle_graph, hyper_paddle
        >>> diamond = HyperGraph(bicycle_graph().incidence).to_simplegraph()
        >>> type(diamond)
        <class 'stochastic_matching.graphs.classes.SimpleGraph'>
        >>> diamond.adjacency
        array([[0, 1, 1, 0],
               [1, 0, 1, 1],
               [1, 1, 0, 1],
               [0, 1, 1, 0]])
        >>> candy = hyper_paddle()
        >>> candy.to_simplegraph()
        Traceback (most recent call last):
        ...
        ValueError: The incidence matrix does not correspond to a simple graph.
        """
        return SimpleGraph(adjacency=incidence_to_adjacency(self.incidence), names=self.names)

    def vis_inputs(self, options=None, nodes_dict=None, edges_dict=None):
        """
        The method provides a Vis-ready description of the graph.

        Parameters
        ----------
        options: :class:`dict`
            Additional / overriding options to pass to the vis engine.
            One specific key, *bipartite_display*, tells if the bipartite structure should be explicitly shown.
        nodes_dict: :class:`list` of :class:`dict`
            Additional / overriding attributes for the nodes.
        edges_dict: :class:`list` of :class:`dict`
            Additional / overriding attributes for the edges.

        Returns
        -------
        :class:`tuple`
            Inputs for the vis engine.

        Examples
        ---------

        >>> from stochastic_matching.graphs.generators import hyper_paddle
        >>> hyper_paddle().vis_inputs() # doctest: +NORMALIZE_WHITESPACE
        ([{'id': 0, 'label': '0', 'title': '0', 'x': 0, 'group': 'Node'},
        {'id': 1, 'label': '1', 'title': '1', 'x': 0, 'group': 'Node'},
        {'id': 2, 'label': '2', 'title': '2', 'x': 0, 'group': 'Node'},
        {'id': 3, 'label': '3', 'title': '3', 'x': 0, 'group': 'Node'},
        {'id': 4, 'label': '4', 'title': '4', 'x': 0, 'group': 'Node'},
        {'id': 5, 'label': '5', 'title': '5', 'x': 0, 'group': 'Node'},
        {'id': 6, 'label': '6', 'title': '6', 'x': 0, 'group': 'Node'},
        {'id': 7, 'title': '0: (0, 1)', 'group': 'HyperEdge', 'x': 480},
        {'id': 8, 'title': '1: (0, 2)', 'group': 'HyperEdge', 'x': 480},
        {'id': 9, 'title': '2: (1, 2)', 'group': 'HyperEdge', 'x': 480},
        {'id': 10, 'title': '3: (4, 5)', 'group': 'HyperEdge', 'x': 480},
        {'id': 11, 'title': '4: (4, 6)', 'group': 'HyperEdge', 'x': 480},
        {'id': 12, 'title': '5: (5, 6)', 'group': 'HyperEdge', 'x': 480},
        {'id': 13, 'title': '6: (2, 3, 4)', 'group': 'HyperEdge', 'x': 480}],
        [{'from': 0, 'to': 7, 'title': '0 <-> 0: (0, 1)'},
        {'from': 0, 'to': 8, 'title': '0 <-> 1: (0, 2)'},
        {'from': 1, 'to': 7, 'title': '1 <-> 0: (0, 1)'},
        {'from': 1, 'to': 9, 'title': '1 <-> 2: (1, 2)'},
        {'from': 2, 'to': 8, 'title': '2 <-> 1: (0, 2)'},
        {'from': 2, 'to': 9, 'title': '2 <-> 2: (1, 2)'},
        {'from': 2, 'to': 13, 'title': '2 <-> 6: (2, 3, 4)'},
        {'from': 3, 'to': 13, 'title': '3 <-> 6: (2, 3, 4)'},
        {'from': 4, 'to': 10, 'title': '4 <-> 3: (4, 5)'},
        {'from': 4, 'to': 11, 'title': '4 <-> 4: (4, 6)'},
        {'from': 4, 'to': 13, 'title': '4 <-> 6: (2, 3, 4)'},
        {'from': 5, 'to': 10, 'title': '5 <-> 3: (4, 5)'},
        {'from': 5, 'to': 12, 'title': '5 <-> 5: (5, 6)'},
        {'from': 6, 'to': 11, 'title': '6 <-> 4: (4, 6)'},
        {'from': 6, 'to': 12, 'title': '6 <-> 5: (5, 6)'}],
        {'groups': {'HyperEdge': {'fixed': {'x': False}, 'color': {'background': 'black'}, 'shape': 'dot', 'size': 5},
        'Node': {'fixed': {'x': False}}},
        'bipartite_display': False})
        """
        if options is None:
            options = dict()

        vis_options = {**HYPER_GRAPH_VIS_OPTIONS, **options}
        value = vis_options['bipartite_display']
        vis_options['groups']['HyperEdge']['fixed']['x'] = value
        vis_options['groups']['Node']['fixed']['x'] = value

        inner_width = round(.8*vis_options.get('width', 600))

        vis_nodes = [{'id': i,
                      'label': self.int_2_str(i),
                      'title': f"{i}: {self.int_2_str(i)}" if self.names is not None else str(i),
                      'x': 0, 'group': 'Node'} for i in range(self.n)]
        if nodes_dict is not None:
            vis_nodes = [{**internal, **external} for internal, external in zip(vis_nodes, nodes_dict)]

        vis_edges = [{'id': self.n + j,
                      'title': f"{j}: ({', '.join([self.int_2_str(i) for i in neighbors(j, self.co_incidence)])})",
                      'group': 'HyperEdge', 'x': inner_width} for j in range(self.m)]
        if edges_dict is not None:
            vis_edges = [{**internal, **external} for internal, external in zip(vis_edges, edges_dict)]

        vis_links = [{'from': i, 'to': self.n + int(j),
                      'title': f"{self.int_2_str(i)} <-> {vis_edges[j]['title']}"} for i in range(self.n)
                     for j in neighbors(i, self.incidence)]
        return vis_nodes + vis_edges, vis_links, vis_options


class SimpleGraph(GenericGraph):
    """
    Parameters
    ----------
    incidence: :class:`~numpy.ndarray`
        Incidence matrix
    adjacency: :class:`~numpy.ndarray`, optional
        Adjacency matrix. Can be used instead of incidence (setting it automatically adjusts the other attributes).

    Attributes
    ----------
    n: :class:`int`
        Number of nodes.
    m: :class:`int`
        Number of edges.
    incidence: :class:`~scipy.sparse.csr_matrix`
        :class:`~scipy.sparse.csr_matrix` view of the incidence matrix.
    co_incidence: :class:`~scipy.sparse.csc_matrix`
        :class:`~scipy.sparse.csc_matrix` view of the incidence matrix.
    names: :class:`list` of :class:`str` or 'alpha', optional
        List of node names (e.g. for display)
    """
    def __init__(self, adjacency=None, incidence=None, names=None):
        super().__init__(incidence=incidence, names=names)
        self.adjacency = adjacency

    @property
    def adjacency(self):
        return self.__adjacency

    @adjacency.setter
    def adjacency(self, adjacency):
        self.__adjacency = adjacency
        if adjacency is not None:
            self.incidence = adjacency_to_incidence(adjacency)

    def to_hypergraph(self):
        """
        Converts to hypergraph.

        Returns
        -------
        :class:`~stochastic_matching.graphs.classes.HyperGraph`

        Examples
        --------
        >>> from stochastic_matching.graphs.generators import bicycle_graph, hyper_paddle
        >>> diamond = bicycle_graph().to_hypergraph()
        >>> type(diamond)
        <class 'stochastic_matching.graphs.classes.HyperGraph'>
        >>> diamond.incidence.toarray().astype('int')
        array([[1, 1, 0, 0, 0],
               [1, 0, 1, 1, 0],
               [0, 1, 1, 0, 1],
               [0, 0, 0, 1, 1]])

        Remind that HyperGraph do not have an adjacency matrix.
        >>> diamond.adjacency
        Traceback (most recent call last):
        ...
        AttributeError: 'HyperGraph' object has no attribute 'adjacency'
        """
        return HyperGraph(incidence=self.incidence, names=self.names)

    def vis_inputs(self, options=None, nodes_dict=None, edges_dict=None):
        """
        The method provides a Vis-ready description of the graph.

        Parameters
        ----------
        options: :class:`dict`
            Additional / overriding options to pass to the vis engine.
        nodes_dict: :class:`list` of :class:`dict`
            Additional / overriding attributed for the nodes.
        edges_dict: :class:`list` of :class:`dict`
            Additional / overriding attributed for the edges.

        Returns
        -------
        :class:`tuple`
            Inputs for the vis engine.

        Examples
        ---------

        >>> from stochastic_matching.graphs.generators import tadpole_graph
        >>> tadpole_graph().vis_inputs() # doctest: +NORMALIZE_WHITESPACE
        ([{'id': 0, 'label': '0', 'title': '0'},
        {'id': 1, 'label': '1', 'title': '1'},
        {'id': 2, 'label': '2', 'title': '2'},
        {'id': 3, 'label': '3', 'title': '3'}],
        [{'from': 0, 'to': 1, 'title': '0: (0, 1)', 'label': '(0, 1)'},
        {'from': 0, 'to': 2, 'title': '1: (0, 2)', 'label': '(0, 2)'},
        {'from': 1, 'to': 2, 'title': '2: (1, 2)', 'label': '(1, 2)'},
        {'from': 2, 'to': 3, 'title': '3: (2, 3)', 'label': '(2, 3)'}],
        {})

        Nodes can have names.

        >>> tadpole_graph(names=['One', 'Two', 'Three', 'Four']).vis_inputs() # doctest: +NORMALIZE_WHITESPACE
        ([{'id': 0, 'label': 'One', 'title': '0: One'},
        {'id': 1, 'label': 'Two', 'title': '1: Two'},
        {'id': 2, 'label': 'Three', 'title': '2: Three'},
        {'id': 3, 'label': 'Four', 'title': '3: Four'}],
        [{'from': 0, 'to': 1, 'title': '0: (One, Two)', 'label': '(One, Two)'},
        {'from': 0, 'to': 2, 'title': '1: (One, Three)', 'label': '(One, Three)'},
        {'from': 1, 'to': 2, 'title': '2: (Two, Three)', 'label': '(Two, Three)'},
        {'from': 2, 'to': 3, 'title': '3: (Three, Four)', 'label': '(Three, Four)'}],
        {})

        Pass 'alpha' to name for automatic letter labeling.

        >>> tadpole_graph(names='alpha').vis_inputs() # doctest: +NORMALIZE_WHITESPACE
        ([{'id': 0, 'label': 'A', 'title': '0: A'},
        {'id': 1, 'label': 'B', 'title': '1: B'},
        {'id': 2, 'label': 'C', 'title': '2: C'},
        {'id': 3, 'label': 'D', 'title': '3: D'}],
        [{'from': 0, 'to': 1, 'title': '0: (A, B)', 'label': '(A, B)'},
        {'from': 0, 'to': 2, 'title': '1: (A, C)', 'label': '(A, C)'},
        {'from': 1, 'to': 2, 'title': '2: (B, C)', 'label': '(B, C)'},
        {'from': 2, 'to': 3, 'title': '3: (C, D)', 'label': '(C, D)'}],
        {})
        """
        if options is None:
            options = dict()
        vis_nodes = [{'id': i, 'label': self.int_2_str(i),
                      'title': f"{i}: {self.int_2_str(i)}" if self.names is not None else str(i)}
                       for i in range(self.n)]
        if nodes_dict is not None:
            vis_nodes = [{**internal, **external} for internal, external in zip(vis_nodes, nodes_dict)]

        vis_edges = [{'from': int(e[0]), 'to': int(e[1]),
                      'title': f"{j}: ({', '.join([self.int_2_str(i) for i in e])})",
                      'label': f"({', '.join([self.int_2_str(i) for i in e])})"}
                     for j, e in [(j, neighbors(j, self.co_incidence)) for j in range(self.m)]]
        if edges_dict is not None:
            vis_edges = [{**internal, **external} for internal, external in zip(vis_edges, edges_dict)]

        return vis_nodes, vis_edges, options
