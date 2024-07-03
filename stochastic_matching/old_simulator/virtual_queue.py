import numpy as np
from numba import njit

from stochastic_matching.old_simulator.generic import Simulator


@njit(cache=True)
def vq_core(prob, alias, number_events, seed,
            incid_ptr, incid_ind, coinc_ptr, coinc_ind,
            scores, trafic, queue_log, steps_done, forbidden_edges):
    """
    Core virtual queue old_simulator. Currently fully monobloc for performance.

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
    scores: :class:`~numpy.ndarray`
        Scores of edges.
    trafic: :class:`~numpy.ndarray`
        Monitor trafic on edges.
    queue_log: :class:`~numpy.ndarray`
        Monitor queue sizes.
    steps_done: :class:`int`
        Number of arrivals processed so far.
    forbidden_edges: :class:`list`
        Edges that are disabled.


    Returns
    -------
    :class:`int`
        Number of steps processed.
    """

    # Retrieve number of nodes and max_queue
    n, max_queue = queue_log.shape
    # Retrieve number of edges
    m = len(trafic)

    ready_edges = np.zeros(m, dtype=np.bool_)
    tails = np.array([-1-e for e in range(m)])
    vq = {t: number_events for t in tails}
    queue_size = np.zeros(n, dtype=np.uint32)

    # Initiate random generator if seed is given
    if seed is not None:
        np.random.seed(seed)

    if forbidden_edges is not None:
        forbid = {k: True for k in forbidden_edges}
        for k in forbid:
            scores[k] = -1

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
            if forbidden_edges is None or e not in forbid:
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
            # add edge to virtual queue
            vq[tails[e]] = age # tail points to a real age
            vq[age] = number_events # new age points to infinity
            tails[e] = age # Move up tail
            # Virtual pop of items:
            # for each node of the edge, lower all adjacent edges by one
            for i in coinc_ind[coinc_ptr[e]:coinc_ptr[e + 1]]:
                scores[incid_ind[incid_ptr[i]:incid_ptr[i + 1]]] -= 1

        # Can a virtual edge be popped?
        earliest = number_events
        e = 0
        # Take oldest activable edge
        for ee in range(m):
            if ready_edges[ee] and vq[-1-ee] < earliest:
                e, earliest = ee, vq[-1-ee]
        # If one exists, starts working
        if earliest < number_events:
            vq[-1-e] = vq[earliest]
            del vq[earliest]
            if earliest == tails[e]:
                tails[e] = -1-e
            trafic[e] += 1  # Update trafic
            for i in coinc_ind[coinc_ptr[e]:coinc_ptr[e + 1]]:
                queue_size[i] -= 1  # Update physical queue sizes
                if queue_size[i] == 0:  # Check queue exhaustion
                    for f in incid_ind[incid_ptr[i]:incid_ptr[i + 1]]:
                        ready_edges[f] = False
    return steps_done + age + 1  # Return the updated number of steps achieved.


class VQSimulator(Simulator):
    """
    Non-Greedy Matching old_simulator derived from :class:`~stochastic_matching.old_simulator.generic.Simulator`.
    Always pick-up the best edge according to a scoring function, even if that edge cannot be used (yet).

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    weights: :class:`~numpy.ndarray`, optional
        Target rewards on edges.
    beta: :class:`float`
        Stabilization parameter. Close to 0, reward maximization is better but queues are more full.
    forbidden_edges: :class:`list`, optional
        Edges that are disabled.
    **kwargs
        Keyword arguments.


    Examples
    --------

    Let start with a working triangle. One can notice the results are different from the ones common to all
    greedy old_simulator.

    >>> import stochastic_matching as sm
    >>> sim = VQSimulator(sm.Cycle(rates=[3, 4, 5]), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint64),
    'queue_log': array([[837, 105,  41,  13,   3,   1,   0,   0,   0,   0],
           [784, 131,  53,  22,   8,   2,   0,   0,   0,   0],
           [629, 187,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint64),
    'steps_done': 1000}

    Unstable diamond (simulation ends before completion due to drift).

    >>> sim = VQSimulator(sm.CycleChain(rates='uniform'), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([34, 42,  7, 41, 36], dtype=uint64),
    'queue_log': array([[142,  55,  34,  25,  23,  16,  24,  12,   7,   1],
           [319,  16,   3,   1,   0,   0,   0,   0,   0,   0],
           [317,  17,   4,   1,   0,   0,   0,   0,   0,   0],
           [107,  67,  64,  37,  22,  24,   7,   3,   6,   2]], dtype=uint64),
    'steps_done': 339}

    Stable diamond without reward optimization:

    >>> sim = VQSimulator(sm.CycleChain(rates=[1, 2, 2, 1]), number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([ 95,  84, 161,  84,  75], dtype=uint64),
    'queue_log': array([[823, 120,  38,  19,   0,   0,   0,   0,   0,   0],
           [626, 215,  75,  46,  28,   9,   1,   0,   0,   0],
           [686, 212,  70,  24,   7,   1,   0,   0,   0,   0],
           [823, 118,  39,  12,   4,   4,   0,   0,   0,   0]], dtype=uint64),
    'steps_done': 1000}

    Let's optimize (kill traffic on first and last edges).

    >>> sim = VQSimulator(sm.CycleChain(rates=[1, 2, 2, 1]), weights=[0, 1, 1, 1, 0], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([ 0, 22, 24, 27,  0], dtype=uint64),
    'queue_log': array([[137,  16,   3,   0,   0,   0,   0,   0,   0,   0],
           [113,  14,   8,   6,   7,   4,   3,   1,   0,   0],
           [ 74,  21,  13,   7,   9,   8,   4,   5,  12,   3],
           [106,  31,  11,   5,   2,   1,   0,   0,   0,   0]], dtype=uint64),
    'steps_done': 156}

    OK, it's working but we reached the maximal queue size quite fast. Let's reduce the pressure.

    >>> sim = VQSimulator(sm.CycleChain(rates=[1, 2, 2, 1]), weights=[0, 1, 1, 1, 0], beta=.8, number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([ 32, 146, 161, 146,  13], dtype=uint64),
    'queue_log': array([[692, 221,  68,  18,   1,   0,   0,   0,   0,   0],
           [515, 153, 123, 109,  48,  32,  16,   4,   0,   0],
           [662, 136,  91,  62,  38,   4,   2,   2,   2,   1],
           [791, 128,  50,  18,   7,   6,   0,   0,   0,   0]], dtype=uint64), 'steps_done': 1000}

    A stable candy. While candies are not good for greedy policies, the virtual queue is
    designed to deal with it.

    >>> sim = VQSimulator(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([109,  29,  17,  59,  58,  62, 107], dtype=uint64),
    'queue_log': array([[302,  83,  93,  83,  54,  60,  43,  48,  41,  32,  49,  60,  31,
              3,   8,   7,   3,   0,   0,   0,   0,   0,   0,   0,   0],
           [825, 101,  27,  14,  26,   7,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [900,  63,  20,   8,   9,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [111,  49,  93, 108, 120, 101,  48,  28,  53,  32,   7,  30,  16,
             42,  28,  34,  31,  30,  26,  11,   2,   0,   0,   0,   0],
           [276, 185, 151,  95,  60,  40,  49,  52,  47,  33,  12,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [755, 142,  67,  36,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [710, 228,  47,  12,   3,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint64), 'steps_done': 1000}

    Last but not least: Stolyar's example from https://arxiv.org/abs/1608.01646

    >>> stol = sm.Model(incidence=[[1, 0, 0, 0, 1, 0, 0],
    ...                       [0, 1, 0, 0, 1, 1, 1],
    ...                       [0, 0, 1, 0, 0, 1, 1],
    ...                       [0, 0, 0, 1, 0, 0, 1]], rates=[1.2, 1.5, 2, .8])

    Without optimization, all we do is self-matches (no queue at all):

    >>> sim = VQSimulator(stol, number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs['trafic']
    array([236, 279, 342, 143,   0,   0,   0], dtype=uint64)
    >>> sim.compute_average_queues()
    array([0., 0., 0., 0.])

    >>> rewards = [-1, -1, 1, 2, 5, 4, 7]

    With optimization, we get the desired results at the price of a huge queue:

    >>> sim = VQSimulator(stol, weights=rewards, number_events=3000, seed=42, max_queue=300)
    >>> sim.run()
    >>> sim.logs['trafic']
    array([  0,   0, 761, 242, 591,   0, 205], dtype=uint64)
    >>> sim.compute_average_queues()
    array([86.13933333,  0.53233333, 92.99666667,  0.33533333])

    Same trafic could be achieved with much lower queues by enforcing forbidden edges:

    >>> sim = VQSimulator(stol, forbidden_edges=[0, 1, 5], number_events=3000, seed=42, max_queue=300)
    >>> sim.run()
    >>> sim.logs['trafic']
    array([  0,   0, 961, 342, 691,   0, 105], dtype=uint64)
    >>> sim.compute_average_queues()
    array([2.731     , 0.69633333, 0.22266667, 0.052     ])
    """

    name = 'virtual_queue'
    """
    String that can be used to refer to that old_simulator.
    """

    def __init__(self, model, weights=None, beta=.01, forbidden_edges=None, **kwargs):
        self.weight = np.array(weights) if weights is not None else None
        self.beta = beta
        self.forbidden_edges = forbidden_edges
        super(VQSimulator, self).__init__(model, **kwargs)

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
        self.inners['scores'] = np.zeros(self.model.m) if self.weight is None else self.weight/self.beta
        self.inners['forbidden_edges'] = self.forbidden_edges

    def set_core(self):
        """
        Plug in the virtual queue core engine.

        Returns
        -------
        None
        """
        self.core = vq_core
