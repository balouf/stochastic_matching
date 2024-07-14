from numba import njit
import numpy as np

from stochastic_matching.simulator.extended import ExtendedSimulator


@njit
def longest_core(logs, arrivals, graph, n_steps, queue_size,  # Generic arguments
                 scores, forbidden_edges, k):
    """
    Jitted function for policies based on longest-queue first.

    Parameters
    ----------
    logs: :class:`~stochastic_matching.simulator.logs.Logs`
        Monitored variables.
    arrivals: :class:`~stochastic_matching.simulator.arrivals.Arrivals`
        Item arrival process.
    graph: :class:`~stochastic_matching.simulator.graph.JitHyperGraph`
        Model graph.
    n_steps: :class:`int`
        Number of arrivals to process.
    queue_size: :class:`~numpy.ndarray`
        Number of waiting items of each class.
    scores: :class:`~numpy.ndarray`
        Default value for edges. Enables EGPD-like selection mechanism.
    forbidden_edges: :class:`list`, optional
        Edges that are disabled.
    k: :class:`int`, optional
        Queue size above which forbidden edges become available again.

    Returns
    -------
    None
    """

    n, max_queue = logs.queue_log.shape

    # Optimize forbidden edges and set greedy flag.
    greedy = False
    if forbidden_edges is not None:
        forbid = {k: True for k in forbidden_edges}
    else:
        forbid = {k: True for k in range(0)}
        if np.min(scores) > -1:
            greedy = True

    for age in range(n_steps):

        # Draw an arrival
        node = arrivals.draw()
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return None

        # Test if an actionable edge may be present
        best_edge = -1
        if not greedy or queue_size[node] == 1:
            # Should we activate edge restriction?
            if greedy:
                restrain = False
            else:
                restrain = True
                if k is not None:
                    for v in range(n):
                        if queue_size[v] >= k:
                            restrain = False
                            break

            best_score = 0
            # update scores
            for e in graph.edges(node):
                if restrain and e in forbid:
                    continue
                score = scores[e]
                for v in graph.nodes(e):
                    w = queue_size[v]
                    if w == 0:
                        break
                    if v != node:
                        score += w
                else:
                    if score > best_score:
                        best_edge = e
                        best_score = score

            if best_edge > -1:
                queue_size[graph.nodes(best_edge)] -= 1

        logs.update(queue_size=queue_size, node=node, edge=best_edge)


class Longest(ExtendedSimulator):
    """
    Matching simulator derived from :class:`~stochastic_matching.simulator.extended.ExtendedSimulator`.
    By default, the policy is greedy and whenever multiple edges are actionable,
    the longest queue (sum of queues of adjacent nodes) is chosen.

    Multiple options are available to tweak this behavior.

    Parameters
    ----------

    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    shift_rewards: :class:`bool`, default=True
        Longest only selects edges with a positive score. By default, all actionable edges have a positive score,
        which makes the policy greedy. However, if negative rewards are added into the mix, this is not True anymore.
        Setting shift_rewards to True ensures that default edge scores are non-negative so reward-based selection
        is greedy. Setting it to False enables a non-greedy reward-based selection.
    **kwargs
        Keyword parameters of :class:`~stochastic_matching.simulator.extended.ExtendedSimulator`.

    Examples
    --------

    Let's start with a working triangle. Not that the results are the same for all greedy simulator because
    there are no decision in a triangle (always at most one non-empty queue under a greedy policy).

    >>> import stochastic_matching as sm
    >>> sim = Longest(sm.Cycle(rates=[3, 4, 5]), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [287 338 375]
    Traffic: [125 162 213]
    Queues: [[838 104  41  13   3   1   0   0   0   0]
     [796 119  53  22   8   2   0   0   0   0]
     [640 176  92  51  24   9   5   3   0   0]]
    Steps done: 1000

    A non stabilizable diamond (simulation ends before completion due to drift).

    >>> sim = Longest(sm.CycleChain(rates='uniform'), n_steps=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [85 82 85 86]
    Traffic: [38 38  7 37 40]
    Queues: [[126  74  28  37  21  32  16   1   2   1]
     [326   8   3   1   0   0   0   0   0   0]
     [321  12   4   1   0   0   0   0   0   0]
     [ 90  80  47  37  37  23  11   3   5   5]]
    Steps done: 338

    A stabilizable candy (but candies are not good for greedy policies).

    >>> sim = Longest(sm.HyperPaddle(rates=[1, 1, 1.5, 1, 1.5, 1, 1]), n_steps=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.plogs # doctest: +NORMALIZE_WHITESPACE
    Arrivals: [46 26 32 37 70 35 45]
    Traffic: [24 17  2 23 33 12 13]
    Queues: [[ 23  32  45  38  22  43  31  34  20   3   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [290   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [290   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [ 9   1   7   9   3   3  26  37   4   8  10   9   2  10  40  11   2  16
        3   3  21  27  22   1   7]
     [212  49  22   5   3   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [233  41   6   7   4   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]
     [231  33  16   4   6   1   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0]]
    Steps done: 291

    Using greedy rewards-based longest (insprired by Stolyar EGCD technique), one can try to reach some vertices.

    That works for the Fish graph:

    >>> fish = sm.KayakPaddle(m=4, l=0, rates=[4, 4, 3, 2, 3, 2])
    >>> sim = Longest(fish, rewards=[0, 2, 2, 0, 1, 1, 0], beta=.01, n_steps=10000, seed=42)
    >>> sim.run()

    The flow avoids one edge:

    >>> sim.flow
    array([2.8764, 1.089 , 1.0008, 0.    , 0.8532, 2.079 , 1.0062])

    The price is one node having a big queue:

    >>> avg_queues = sim.avg_queues
    >>> avg_queues[-1]
    61.1767

    Other nodes are not affected:

    >>> np.round(np.mean(avg_queues[:-1]), decimals=4)
    0.7362

    Playing with the beta parameter allows to adjust the trade-off (smaller queue, leak on the forbidden edge):

    >>> sim = Longest(fish, rewards=[0, 2, 2, 0, 1, 1, 0], beta=.1, n_steps=10000, seed=42)
    >>> sim.run()
    >>> sim.flow
    array([2.9574, 1.008 , 0.9198, 0.018 , 0.9972, 2.061 , 1.0242])
    >>> sim.avg_queues[-1]
    8.4628

    Alternatively, one can use the k-filtering techniques:

    >>> diamond = sm.CycleChain(rates=[1, 2, 2, 1])
    >>> diamond.run('longest', forbidden_edges=[0, 4], seed=42,
    ...                            k=100, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([0.   , 0.954, 0.966, 0.954, 0.   ])

    Same result can be achieved by putting low rewards on 0 and 4 and settings forbidden_edges to True.

    >>> diamond.run('longest', rewards=[1, 2, 2, 2, 1], seed=42, forbidden_edges=True,
    ...                            k=100, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([0.   , 0.954, 0.966, 0.954, 0.   ])

    Note that if the threshold is too low some exceptions are done to limit the queue sizes.

    >>> diamond.run('longest', rewards=[1, 2, 2, 2, 1], seed=42, forbidden_edges=True,
    ...                            k=10, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([0.108, 0.93 , 0.966, 0.93 , 0.024])

    Having no relaxation usually leads to large, possibly overflowing, queues.
    However, this doesn't show on small simulations.

    >>> diamond.run('longest', rewards=[1, 2, 2, 2, 1], seed=42, forbidden_edges=True,
    ...                            n_steps=1000, max_queue=100)
    True
    >>> diamond.simulator.avg_queues
    array([7.515, 7.819, 1.067, 1.363])
    >>> diamond.simulation
    array([0.   , 0.954, 0.966, 0.954, 0.   ])

    To compare with the priority-based pure greedy version:

    >>> diamond.run('priority', weights=[1, 2, 2, 2, 1], n_steps=1000, max_queue=1000, seed=42)
    True
    >>> diamond.simulation
    array([0.444, 0.63 , 0.966, 0.63 , 0.324])

    We get similar results with Stolyar greedy longest:

    >>> diamond.run('longest', rewards=[-1, 1, 1, 1, -1], beta=.01, n_steps=1000, max_queue=1000, seed=42)
    True
    >>> diamond.simulation
    array([0.444, 0.63 , 0.966, 0.63 , 0.324])

    However, if you remove the greedyness, you can converge to the vertex:

    >>> diamond.run('longest', rewards=[-1, 1, 1, 1, -1], beta=.01,
    ...             n_steps=1000, max_queue=1000, seed=42, shift_rewards=False)
    True
    >>> diamond.simulation
    array([0.   , 0.954, 0.966, 0.954, 0.   ])

    Another example with other rates.

    >>> diamond.rates=[4, 5, 2, 1]

    Optimize with the first and last edges that provide less reward.

    >>> diamond.run('longest', rewards=[1, 2, 2, 2, 1], seed=42, forbidden_edges=True,
    ...                            k=100, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([3.264, 0.888, 0.948, 0.84 , 0.   ])

    Increase the reward on the first edge.

    >>> diamond.run('longest', rewards=[4, 2, 2, 2, 1], seed=42, forbidden_edges=True,
    ...                            k=100, n_steps=1000, max_queue=1000)
    True
    >>> diamond.simulation
    array([4.152, 0.   , 0.996, 0.   , 0.84 ])

    On bijective graphs, no edge is forbidden whatever the weights.

    >>> paw = sm.Tadpole()
    >>> paw.run('longest', rewards=[6, 3, 1, 2], seed=42, forbidden_edges=True,
    ...                            k=100, n_steps=1000, max_queue=1000)
    True
    >>> paw.simulation
    array([1.048, 1.056, 1.016, 0.88 ])
    """
    name = 'longest'

    def __init__(self, model, shift_rewards=True, **kwargs):
        self.shift_rewards = shift_rewards
        super().__init__(model, **kwargs)

    def set_internal(self):
        super().set_internal()
        if self.shift_rewards:
            self.internal['scores'] -= np.min(self.internal['scores'])

    def run(self):
        longest_core(logs=self.logs, **self.internal)
