import numpy as np
from numba import njit

from stochastic_matching.simulator.generic import Simulator


@njit(cache=True)
def vq_core(prob, alias, number_events, seed,
            incid_ptr, incid_ind, coinc_ptr, coinc_ind,
            ready_edges, scores, vq, queue_size,
            trafic, queue_log, steps_done):
    """
    Core virtual queue simulator. Currently fully monobloc for performance.

    Parameters
    ----------
    prob: :class:`~numpy.ndarray`
        Probabilities to stay in the drawn bucket
    alias: :class:`~numpy.ndarray`
        Redirection array
    number_events: :class:`int`
        Number of arrivals to simulate.
    seed: :class:`int`
        Seed of the random generator
    incid_ptr: :class:`~numpy.ndarray`
        Pointers of the incidence matrix.
    incid_ind: :class:`~numpy.ndarray`
        Indices of the incidence matrix.
    coinc_ptr: :class:`~numpy.ndarray`
        Pointers of the co-incidence matrix.
    coinc_ind: :class:`~numpy.ndarray`
        Indices of the co-incidence matrix.
    ready_edges: :class:`~numpy.ndarray`
        Boolen array of edges physically ready for activation.
    scores: :class:`~numpy.ndarray`
        Scores of edges.
    vq: :class:`~numpy.ndarray`
        Current virtual queue size (can be negative)
    queue_size: :class:`~numpy.ndarray`
        Current queue sizes.
    trafic: :class:`~numpy.ndarray`
        Monitor trafic on edges.
    queue_log: :class:`~numpy.ndarray`
        Monitor queue sizes.
    steps_done: :class:`int`
        Number of arrivals processed so far.

    Returns
    -------
    :class:`int`
        Number of steps processed.
    """

    # Retrieve number of nodes and max_queue
    n, max_queue = queue_log.shape
    # Retrieve number of edges
    m = len(trafic)

    # Initiate random generator if seed is given
    if seed is not None:
        np.random.seed(seed)

    # Start main loop
    age = 0
    for age in range(number_events):

        # Update queue logs
        for j in range(n):
            queue_log[j, queue_size[j]] += 1

        # Draw an arrival
        node = np.random.randint(n)
        if np.random.rand() > prob[node]:
            node = alias[node]

        # Increment queue, deal with overflowing
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return steps_done + age + 1

        # Browse adjacent edges
        for e in incid_ind[incid_ptr[node]:incid_ptr[node + 1]]:
            scores[e] += 1  # Increase score

            # Checks if edge turns to feasible
            if queue_size[node] == 1:
                # noinspection PyUnresolvedReferences
                if np.all(queue_size[coinc_ind[coinc_ptr[e]:coinc_ptr[e + 1]]] > 0):
                    ready_edges[e] = True

        # Select best edge
        # noinspection PyUnresolvedReferences
        e = np.argmax(scores)

        # If the best edge is worthy
        if scores[e] > 0:
            vq[e] += 1  # Add edge to virtual queue

            # Virtual pop of items:
            # for each node of the edge, lower all adjacent edges by one
            for i in coinc_ind[coinc_ptr[e]:coinc_ptr[e + 1]]:
                scores[incid_ind[incid_ptr[i]:incid_ptr[i + 1]]] -= 1

        for e in range(m):
            # Can a virtual edge be popped?
            if ready_edges[e] and vq[e] > 0:
                vq[e] -= 1  # Pop from virtual queue
                trafic[e] += 1  # Update trafic
                for i in coinc_ind[coinc_ptr[e]:coinc_ptr[e + 1]]:
                    queue_size[i] -= 1  # Update physical queue sizes
                    if queue_size[i] == 0:  # Check queue exhaustion
                        for f in incid_ind[incid_ptr[i]:incid_ptr[i + 1]]:
                            ready_edges[f] = False
                break  # Uncomment the break would allow multiple pops per turn.

    return steps_done + age + 1  # Return the updated number of steps achieved.


class VQSimulator(Simulator):
    """
    Non-Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.generic.Simulator`.
    Always pick-up the best edge according to a scoring function, even if that edge cannot be used (yet).

    Examples
    --------

    Let start with a working triangle. One can notice the results are different from the ones common to all
    greedy simulator.

    >>> import stochastic_matching as sm
    >>> sim = VQSimulator(sm.Cycle(rates=[3, 4, 5]), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint64),
    'queue_log': array([[836, 106,  41,  13,   3,   1,   0,   0,   0,   0],
           [788, 128,  52,  22,   8,   2,   0,   0,   0,   0],
           [623, 186,  96,  54,  24,   9,   5,   3,   0,   0]], dtype=uint64),
    'steps_done': 1000}

    Unstable diamond (simulation ends before completion due to drift).

    >>> sim = VQSimulator(sm.CycleChain(rates='uniform'), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([35, 43,  7, 39, 34], dtype=uint64),
    'queue_log': array([[156,  68,  56,  34,  14,   1,   0,   0,   0,   0],
           [306,  19,   3,   1,   0,   0,   0,   0,   0,   0],
           [306,  18,   4,   1,   0,   0,   0,   0,   0,   0],
           [ 98,  67,  35,  25,  26,  30,  16,  11,  10,  11]], dtype=uint64),
    'steps_done': 329}

    A stable candy. While candies are not good for greedy policies, the virtual queue is
    designed to deal with it.

    >>> sim = VQSimulator(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([109,  29,  17,  59,  58,  62, 107], dtype=uint64),
    'queue_log': array([[302,  85,  97,  94,  65,  53,  60,  45,  36,  46,  59,  45,  10,
              3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [839, 102,  12,  29,  18,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [884,  79,  20,   8,   9,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [113,  44,  82, 101,  96,  99,  67,  31,  29,  50,  14,  20,  32,
             31,  44,  44,  34,  30,  26,  11,   2,   0,   0,   0,   0],
           [239, 154, 138, 103,  75,  69,  58,  67,  52,  33,  12,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [755, 143,  72,  30,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [709, 229,  48,  13,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint64),
    'steps_done': 1000}
    """

    name = 'virtual_queue'
    """
    String that can be use to refer to that simulator.
    """

    def set_inners(self):
        """
        Defines inner variables for the virtual queue core engine.

        Returns
        -------
        None
        """
        self.inners = dict()
        self.inners['incid_ptr'] = self.model.incidence_csr.indptr
        self.inners['incid_ind'] = self.model.incidence_csr.indices
        self.inners['coinc_ptr'] = self.model.incidence_csc.indptr
        self.inners['coinc_ind'] = self.model.incidence_csc.indices
        self.inners['ready_edges'] = np.zeros(self.model.m, dtype=bool)
        self.inners['scores'] = np.zeros(self.model.m, dtype=np.int32)
        self.inners['vq'] = np.zeros(self.model.m, dtype=np.uint32)
        self.inners['queue_size'] = np.zeros(self.model.n, dtype=np.uint32)

    def set_core(self):
        """
        Plug in the virtual queue core engine.

        Returns
        -------
        None
        """
        self.core = vq_core
