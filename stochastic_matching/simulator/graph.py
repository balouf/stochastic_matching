from numba import int32
from numba.experimental import jitclass


def make_jit_graph(model):
    """
    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model that contains an incidence matrix.

    Returns
    -------
    :class:`~stochastic_matching.simulator.graph.JitHyperGraph`
        Jitted graph.
    """
    return JitHyperGraph(incid_ind=model.incidence_csr.indices, incid_ptr=model.incidence_csr.indptr,
                         coinc_ind=model.incidence_csc.indices, coinc_ptr=model.incidence_csc.indptr,
                         n=model.n, m=model.m)


@jitclass
class JitHyperGraph:
    """
    Jit compatible view of a (hyper)graph.

    Parameters
    ----------
    incid_ind: :class:`~numpy.ndarray`
        Indices of the incidence matrix.
    incid_ptr: :class:`~numpy.ndarray`
        Pointers of the incidence matrix.
    coinc_ind: :class:`~numpy.ndarray`
        Indices of the co-incidence matrix.
    coinc_ptr: :class:`~numpy.ndarray`
        Pointers of the co-incidence matrix.
    n: :class:`int`
        Number of nodes.
    m: :class:`int`
        Number of (hyper)edges.
    """
    incid_ind: int32[:]
    incid_ptr: int32[:]
    coinc_ind: int32[:]
    coinc_ptr: int32[:]
    n: int
    m: int

    def __init__(self, incid_ind, incid_ptr, coinc_ind, coinc_ptr, n, m):
        self.incid_ind = incid_ind
        self.incid_ptr = incid_ptr
        self.coinc_ind = coinc_ind
        self.coinc_ptr = coinc_ptr
        self.n = n
        self.m = m

    def edges(self, node):
        """
        Parameters
        ----------
        node: :class:`int`
            Index of a node.

        Returns
        -------
        :class:`~numpy.ndarray`
            Edges adjacent to the node.
        """
        return self.incid_ind[self.incid_ptr[node]:self.incid_ptr[node + 1]]

    def nodes(self, edge):
        """
        Parameters
        ----------
        edge: :class:`int`
            Index of an edge.

        Returns
        -------
        :class:`~numpy.ndarray`
            Nodes adjacent to the edge.
        """
        return self.coinc_ind[self.coinc_ptr[edge]:self.coinc_ptr[edge + 1]]
