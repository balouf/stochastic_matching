import numpy as np
from numba import njit


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
