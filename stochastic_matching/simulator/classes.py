from abc import ABC

import numpy as np
from matplotlib import pyplot as plt

from stochastic_matching.simulator.common import create_prob_alias, graph_neighbors_list
from stochastic_matching.simulator.virtual_queue import vq_core
from stochastic_matching.simulator.greedy import qs_core_maker, random_node_selector, longest_queue_selector, \
    longest_sum_queue_selector, random_item_selector, random_sum_item_selector, qstate_core_maker, fcfm_selector, \
    fcfm_hyper_selector


class Simulator:
    """
    Abstract class that describes the generic behavior of matching queues simulator. See sub-classes for examples.

    Parameters
    ----------
    graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graphs.classes.HyperGraph`
        Input graph.
    mu: :class:`~numpy.ndarray` or :class:`list`
        Arrival rates on nodes of the graph.
    number_events: :class:`int`, optional
        Number of arrivals to simulate.
    seed: :class:`int`, optional
        Seed of the random generator
    max_queue: :class:`int`
        Max queue size. Necessary for speed and detection of unstability.
        For stable systems very close to the unstability
        border, the max_queue may be reached.

    Attributes
    ----------
    generator: :class:`dict`
        Generator parameters (prob and alias vector, seed, and number of events).
    inners: :class:`dict`
        Inner variable (depends on the exact simulator engine used).
    logs: :class:`dict`
        Monitored variables (default to trafic on edges,
        queue size distribution, and number of steps achieved).
    core: callable
        Core simulator, usually a numba function. Must return the total number of steps achieved.
    """

    name = None
    """
    Name that can be used to list all non-abstract classes.
    """

    def __init__(self, graph, mu, number_events=1000000, seed=None, max_queue=1000):
        self.graph = graph
        self.mu = np.array(mu)
        self.max_queue = max_queue

        self.generator = None
        self.set_generator(mu, number_events, seed)

        self.inners = None
        self.set_inners()

        self.logs = None
        self.set_logs()

        self.core = None
        self.set_core()

    def set_generator(self, mu, number_events, seed):
        """
        Populate the generator parameters.

        Parameters
        ----------
        mu: :class:`~numpy.ndarray` or :class:`list`
            Arrival rates on nodes of the graph.
        number_events: :class:`int`, optional
            Number of arrivals to simulate.
        seed: :class:`int`, optional
            Seed of the random generator

        Returns
        -------
        None
        """
        prob, alias = create_prob_alias(mu)
        self.generator = {'prob': prob, 'alias': alias,
                          'number_events': number_events, 'seed': seed}

    def set_inners(self):
        """
        Populate the inner parameters.

        Returns
        -------
        None
        """
        self.inners = {'neighbors': graph_neighbors_list(self.graph),
                       }

    def set_logs(self):
        """
        Populate the monitored variables.

        Returns
        -------

        """
        self.logs = {'trafic': np.zeros(self.graph.m, dtype=np.uint32),
                     'queue_log': np.zeros((self.graph.n, self.max_queue), dtype=np.uint32),
                     'steps_done': 0}

    def set_core(self, **kwargs):
        raise NotImplementedError

    def reset(self):
        """
        Reset inner and monitored variables.

        Returns
        -------
        None
        """
        self.set_inners()
        self.set_logs()

    def run(self):
        """
        Run simulation (results are stored in the attribute :attr:`~stochastic_matching.simulator.classes.Simulator.logs`).

        Returns
        -------
        """
        self.logs['steps_done'] = self.core(**self.generator,
                                            **self.inners, **self.logs)

    def compute_average_queues(self):
        """
        Returns
        -------
        :class:`~numpy.ndarray`
            Average queue sizes.

        Examples
        --------

        >>> from stochastic_matching.graphs.generators import bicycle_graph
        >>> sim = FCFM(bicycle_graph(), [2, 2.1, 1.1, 1], seed=42, number_events=1000, max_queue=8)
        >>> sim.run()
        >>> sim.compute_average_queues()
        array([1.07826087, 0.26086957, 0.07391304, 0.85217391])
        """
        return self.logs['queue_log'].dot(np.arange(self.max_queue)) / self.logs['steps_done']

    def total_waiting_time(self):
        """
        Returns
        -------
        :class:`float`
            Average waiting time

        Examples
        --------

        >>> from stochastic_matching.graphs.generators import bicycle_graph
        >>> sim = FCFM(bicycle_graph(), [2, 2.1, 1.1, 1], seed=42, number_events=1000, max_queue=8)
        >>> sim.run()
        >>> sim.total_waiting_time()
        0.36535764375876584
        """
        return np.sum(self.compute_average_queues())/np.sum(self.mu)

    def show_average_queues(self, indices=None, sort=False, as_time=False):
        """
        Parameters
        ----------
        indices: :class:`list`, optional
            Indices of the nodes to display
        sort: :class:`bool`, optional
            If True, display the nodes by decreasing average queue size
        as_time: :class:`bool`, optional
            If True, display the nodes by decreasing average queue size

        Returns
        -------
        :class:`~matplotlib.figure.Figure`
            A figure of the CCDFs of the queues.

        Examples
        --------

        >>> from stochastic_matching.graphs.generators import bicycle_graph
        >>> sim = FCFM(bicycle_graph(), [2, 2.1, 1.1, 1], seed=42, number_events=1000, max_queue=8)
        >>> sim.run()

        On IPython, this will draw the CCDF, otherwise a fig object is returned.

        >>> fig = sim.show_average_queues()
        >>> fig #doctest: +ELLIPSIS
        <Figure size ...x... with 1 Axes>

        >>> fig = sim.show_average_queues(indices=[0, 3, 2], sort=True, as_time=True)
        >>> fig #doctest: +ELLIPSIS
        <Figure size ...x... with 1 Axes>
        """
        averages = self.compute_average_queues()
        if as_time:
            averages = averages / self.mu
        if indices is not None:
            averages = averages[indices]
            names = [self.graph.int_2_str(i) for i in indices]
        else:
            names = [self.graph.int_2_str(i) for i in range(self.graph.n)]
        if sort is True:
            ind = np.argsort(-averages)
            averages = averages[ind]
            names = [names[i] for i in ind]
        plt.bar(names, averages)
        if as_time:
            plt.ylabel("Average waiting time")
        else:
            plt.ylabel("Average queue occupancy")
        plt.xlabel("Node")
        return plt.gcf()

    def compute_ccdf(self):
        """
        Returns
        -------
        :class:`~numpy.ndarray`
            CCDFs of the queues.

        Examples
        --------

        >>> from stochastic_matching.graphs.generators import bicycle_graph
        >>> sim = FCFM(bicycle_graph(), [2, 2.1, 1.1, 1], seed=42, number_events=1000, max_queue=8)
        >>> sim.run()
        >>> sim.compute_ccdf() #doctest: +NORMALIZE_WHITESPACE
        array([[1.        , 0.4826087 , 0.27826087, 0.16521739, 0.1       ,
            0.03913043, 0.00869565, 0.00434783, 0.        ],
           [1.        , 0.17826087, 0.06956522, 0.01304348, 0.        ,
            0.        , 0.        , 0.        , 0.        ],
           [1.        , 0.05217391, 0.0173913 , 0.00434783, 0.        ,
            0.        , 0.        , 0.        , 0.        ],
           [1.        , 0.45217391, 0.23478261, 0.1       , 0.05217391,
            0.01304348, 0.        , 0.        , 0.        ]])
        """

        events = self.logs['steps_done']
        n = self.graph.n
        # noinspection PyUnresolvedReferences
        return (events - np.cumsum(np.hstack([np.zeros((n, 1)), self.logs['queue_log']]), axis=1)) / events

    def show_ccdf(self, indices=None, sort=None):
        """
        Parameters
        ----------
        indices: :class:`list`, optional
            Indices of the nodes to display
        sort: :class:`bool`, optional
            If True, order the nodes by decreasing average queue size

        Returns
        -------
        :class:`~matplotlib.figure.Figure`
            A figure of the CCDFs of the queues.

        Examples
        --------

        >>> from stochastic_matching.graphs.generators import bicycle_graph
        >>> sim = FCFM(bicycle_graph(), [2, 2.1, 1.1, 1], seed=42, number_events=1000, max_queue=8)
        >>> sim.run()

        On IPython, this will draw the CCDF, otherwise a fig object is returned.

        >>> fig = sim.show_ccdf()
        >>> fig #doctest: +ELLIPSIS
        <Figure size ...x... with 1 Axes>

        >>> fig = sim.show_ccdf(indices=[0, 3, 2], sort=True)
        >>> fig #doctest: +ELLIPSIS
        <Figure size ...x... with 1 Axes>
        """
        ccdf = self.compute_ccdf()

        if indices is not None:
            ccdf = ccdf[indices, :]
            names = [self.graph.int_2_str(i) for i in indices]
        else:
            names = [self.graph.int_2_str(i) for i in range(self.graph.n)]
        if sort is True:
            averages = self.compute_average_queues()
            if indices is not None:
                averages = averages[indices]
            ind = np.argsort(-averages)
            ccdf = ccdf[ind, :]
            names = [names[i] for i in ind]
        for i, name in enumerate(names):
            plt.semilogy(ccdf[i, ccdf[i, :] > 0], label=name)
        plt.legend()
        plt.xlim([0, None])
        plt.ylim([None, 1])
        plt.ylabel("CCDF")
        plt.xlabel("Queue occupancy")
        return plt.gcf()


class QueueSizeSimulator(Simulator, ABC):
    """
    Abstract class derived from :class:`~stochastic_matching.simulator.classes.Simulator`
    for greedy simulator based on the size of queues.
    """

    def set_inners(self):
        """
        Incorporate `queue_size` to the inner variables.

        Returns
        -------
        None
        """
        super().set_inners()
        self.inners['queue_size'] = np.zeros(self.graph.n, dtype=np.uint32)

    def set_core_from_selector(self, simple_selector, hyper_selector):
        """
        Sets the core engine for a greedy policy.

        Parameters
        ----------
        simple_selector: callable
            Jitted selector function for simple graphs.
        hyper_selector: callable
            Jitted selector function for hypergraphs.

        Returns
        -------
        None
        """
        self.core = qs_core_maker(self.graph, simple_selector, hyper_selector)


class RandomNode(QueueSizeSimulator):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.classes.QueueSizeSimulator`.
    When multiple choices are possible, one is chosen uniformly at random.

    Examples
    --------

    Let start with a working triangle. One can notice the results are the same for all greedy simulator because
    there are no multiple choices in a triangle (always one non-empty queue at most under a greedy policy).

    >>> from stochastic_matching import tadpole_graph, bicycle_graph, hyper_paddle
    >>> sim = RandomNode(tadpole_graph(n=0), mu=[3, 4, 5], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint32),
     'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
                         [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
                         [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint32),
     'steps_done': 1000}

    Sanity check: results are unchanged if the graph is treated as hypergraph.

    >>> sim = RandomNode(tadpole_graph(n=0).to_hypergraph(), mu=[3, 4, 5], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint32),
     'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
                         [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
                         [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint32),
     'steps_done': 1000}


    A ill braess graph (simulation ends before completion due to drift).

    >>> sim = RandomNode(bicycle_graph(), mu=[1, 1, 1, 1], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([ 7, 10,  1,  4,  7], dtype=uint32),
    'queue_log': array([[22, 13,  7,  7,  5, 15,  4,  0,  0,  0],
           [73,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [69,  3,  1,  0,  0,  0,  0,  0,  0,  0],
           [13, 10, 11,  4,  4,  4,  2,  9, 11,  5]], dtype=uint32),
    'steps_done': 73}

    A working candy (but candies are not good for greedy policies).

    >>> sim = RandomNode(hyper_paddle(), mu=[1, 1, 1.5, 1, 1.5, 1, 1], number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([26, 21,  4, 25, 34, 10, 16], dtype=uint32),
    'queue_log': array([[ 85,  37,  36,  41,  22,  29,  46,  17,   1,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [275,  32,   7,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [313,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 10,   1,   7,   9,   3,   3,  13,   7,  34,  11,   2,   8,  13,
             18,  12,  56,  27,   8,  24,  25,   7,   8,   3,   1,   4],
           [168,  48,  35,  39,  20,   4,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [278,  25,   7,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [268,  23,  16,   6,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint32),
    'steps_done': 314}

    Note that you can reset the simulator before starting another run.

    >>> sim.reset()
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([26, 21,  4, 25, 34, 10, 16], dtype=uint32),
    'queue_log': array([[ 85,  37,  36,  41,  22,  29,  46,  17,   1,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [275,  32,   7,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [313,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 10,   1,   7,   9,   3,   3,  13,   7,  34,  11,   2,   8,  13,
             18,  12,  56,  27,   8,  24,  25,   7,   8,   3,   1,   4],
           [168,  48,  35,  39,  20,   4,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [278,  25,   7,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [268,  23,  16,   6,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint32),
    'steps_done': 314}

    You can display the distribution of queue sizes as a ccdf:

    >>> sim.show_ccdf() # doctest: +ELLIPSIS
    <Figure size ...x... with 1 Axes>
    """

    name = 'random_node'
    """
    String that can be use to refer to that simulator.
    """

    def set_core(self):
        """
        Build the core engine to choose uniformly at random amongst choices.

        Returns
        -------
        None
        """
        self.set_core_from_selector(random_node_selector, random_node_selector)


class LongestQueue(QueueSizeSimulator):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.classes.QueueSizeSimulator`.
    When multiple choices are possible, the longest queue (or sum of queues for hyperedges) is chosen.

    Examples
    --------

    Let start with a working triangle. One can notice the results are the same for all greedy simulator because
    there are no multiple choices in a triangle (always one non-empty queue at most under a greedy policy).

    >>> from stochastic_matching import tadpole_graph, bicycle_graph, hyper_paddle
    >>> sim = LongestQueue(tadpole_graph(n=0), mu=[3, 4, 5], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint32),
    'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
       [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
       [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint32),
    'steps_done': 1000}

    A ill braess graph (simulation ends before completion due to drift).

    >>> sim = LongestQueue(bicycle_graph(), mu=[1, 1, 1, 1], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([38, 38,  7, 37, 40], dtype=uint32),
    'queue_log': array([[127,  74,  28,  37,  21,  32,  16,   1,   2,   1],
           [327,   8,   3,   1,   0,   0,   0,   0,   0,   0],
           [322,  12,   4,   1,   0,   0,   0,   0,   0,   0],
           [ 91,  80,  47,  37,  37,  23,  11,   3,   5,   5]], dtype=uint32),
    'steps_done': 339}

    A working candy (but candies are not good for greedy policies).

    >>> sim = LongestQueue(hyper_paddle(), mu=[1, 1, 1.5, 1, 1.5, 1, 1], number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([24, 17,  2, 23, 33, 12, 13], dtype=uint32),
    'queue_log': array([[ 24,  32,  45,  38,  22,  43,  31,  34,  20,   3,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [291,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [291,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 10,   1,   7,   9,   3,   3,  26,  37,   4,   8,  10,   9,   2,
             10,  40,  11,   2,  16,   3,   3,  21,  27,  22,   1,   7],
           [213,  49,  22,   5,   3,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [234,  41,   6,   7,   4,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [232,  33,  16,   4,   6,   1,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint32),
    'steps_done': 292}
    """

    name = 'longest_queue'
    """
    String that can be use to refer to that simulator.
    """

    def set_core(self):
        """
        Build the core engine to choose the longest (sum of) queue(s).

        Returns
        -------
        None
        """
        self.set_core_from_selector(longest_queue_selector, longest_sum_queue_selector)


class RandomItem(QueueSizeSimulator):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.classes.QueueSizeSimulator`.
    When multiple choices are possible, chooses proportionally to the sizes of the queues
    (or sum of queues for hyperedges).

    Examples
    --------

    Let start with a working triangle. One can notice the results are the same for all greedy simulator because
    there are no multiple choices in a triangle (always one non-empty queue at most under a greedy policy).

    >>> from stochastic_matching import tadpole_graph, bicycle_graph, hyper_paddle
    >>> sim = RandomItem(tadpole_graph(n=0), mu=[3, 4, 5], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint32),
    'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
       [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
       [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint32),
    'steps_done': 1000}

    A ill braess graph (simulation ends before completion due to drift).

    >>> sim = RandomItem(bicycle_graph(), mu=[1, 1, 1, 1], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([12, 11,  4,  8, 10], dtype=uint32),
    'queue_log': array([[39, 13, 10,  6,  3,  8, 14,  8,  3,  1],
           [96,  5,  3,  1,  0,  0,  0,  0,  0,  0],
           [97,  7,  1,  0,  0,  0,  0,  0,  0,  0],
           [41, 18, 13, 13,  8,  5,  1,  2,  3,  1]], dtype=uint32),
    'steps_done': 105}

    A working candy (but candies are not good for greedy policies).

    >>> sim = RandomItem(hyper_paddle(), mu=[1, 1, 1.5, 1, 1.5, 1, 1], number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([83, 62, 36, 58, 75, 48, 74], dtype=uint32),
    'queue_log': array([[537, 135,  65,  62,  34,  20,  25,  30,  48,  12,   4,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [792, 130,  28,  14,   8,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [861,  71,  19,  15,   5,   1,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 10,   1,   7,   9,   3,  31,  65,  70,  46,  56,  60,  82,  59,
             49,  54,  60,  61,  42, 100,  44,  28,  10,  15,   9,   1],
           [711, 127,  77,  34,  19,   4,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [800,  97,  50,  22,   3,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [732, 125,  74,  25,  11,   3,   2,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint32),
    'steps_done': 972}
    """

    name = 'random_item'
    """
    String that can be use to refer to that simulator.
    """

    def set_core(self):
        """
        Build the core engine to choose at random proportionally to the sizes of the (sum of the) queues.

        Returns
        -------
        None
        """
        self.set_core_from_selector(random_item_selector, random_sum_item_selector)


class QueueStateSimulator(Simulator, ABC):
    """
    Abstract class derived from :class:`~stochastic_matching.simulator.classes.Simulator`
    for greedy simulator based on the states of queues (including age of items).
    """

    def set_inners(self):
        """
        Incorporate `queue_start`, `queue_end`, and `items` to the inner variables.

        Returns
        -------
        None
        """
        super().set_inners()
        self.inners['queue_start'] = np.zeros(self.graph.n, dtype=np.uint32)
        self.inners['queue_end'] = np.zeros(self.graph.n, dtype=np.uint32)
        self.inners['items'] = np.zeros((self.graph.n, self.max_queue), dtype=np.uint32)

    def set_core_from_selector(self, simple_selector, hyper_selector):
        """
        Sets the core engine for a greedy policy.

        Parameters
        ----------
        simple_selector: callable
            Jitted selector function for simple graphs.
        hyper_selector: callable
            Jitted selector function for hypergraphs.

        Returns
        -------
        None
        """
        self.core = qstate_core_maker(self.graph, simple_selector, hyper_selector)


class FCFM(QueueStateSimulator):
    """
    Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.classes.QueueStateSimulator`.
    When multiple choices are possible, the oldest item is chosen.

    Examples
    --------

    Let start with a working triangle. One can notice the results are the same for all greedy simulator because
    there are no multiple choices in a triangle (always one non-empty queue at most under a greedy policy).

    >>> from stochastic_matching import tadpole_graph, bicycle_graph, hyper_paddle
    >>> sim = FCFM(tadpole_graph(n=0), mu=[3, 4, 5], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint32),
    'queue_log': array([[838, 104,  41,  13,   3,   1,   0,   0,   0,   0],
       [796, 119,  53,  22,   8,   2,   0,   0,   0,   0],
       [640, 176,  92,  51,  24,   9,   5,   3,   0,   0]], dtype=uint32),
    'steps_done': 1000}

    A ill braess graph (simulation ends before completion due to drift).

    >>> sim = FCFM(bicycle_graph(), mu=[1, 1, 1, 1], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([34, 42,  7, 41, 36], dtype=uint32),
    'queue_log': array([[127,  70,  22,  26,  29,  12,  23,  15,  10,   5],
           [327,   8,   3,   1,   0,   0,   0,   0,   0,   0],
           [322,  12,   4,   1,   0,   0,   0,   0,   0,   0],
           [106,  80,  65,  28,  31,  15,   4,   2,   6,   2]], dtype=uint32),
    'steps_done': 339}

    A working candy (but candies are not good for greedy policies).

    >>> sim = FCFM(hyper_paddle(), mu=[1, 1, 1.5, 1, 1.5, 1, 1],
    ...            number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([24, 17,  2, 23, 33, 12, 13], dtype=uint32),
    'queue_log': array([[ 24,  32,  45,  38,  22,  43,  31,  34,  20,   3,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [291,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [291,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [ 10,   1,   7,   9,   3,   3,  26,  37,   4,   8,  10,   9,   2,
             10,  40,  11,   2,  16,   3,   3,  21,  27,  22,   1,   7],
           [213,  49,  22,   5,   3,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [234,  41,   6,   7,   4,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [232,  33,  16,   4,   6,   1,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
          dtype=uint32),
    'steps_done': 292}
    """

    name = 'fcfm'
    """
    String that can be use to refer to that simulator.
    """

    def set_core(self):
        """
        Build the core engine to choose the oldest item.

        Returns
        -------
        None
        """
        self.set_core_from_selector(fcfm_selector, fcfm_hyper_selector)


class VQSimulator(Simulator):
    """
    Non-Greedy Matching simulator derived from :class:`~stochastic_matching.simulator.classes.Simulator`.
    Always pick-up the best edge according to a scoring function, even if that edge cannot be used (yet).

    Examples
    --------

    Let start with a working triangle. One can notice the results are different from the ones common to all
    greedy simulator.

    >>> from stochastic_matching import tadpole_graph, bicycle_graph, hyper_paddle
    >>> sim = VQSimulator(tadpole_graph(n=0), mu=[3, 4, 5], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([125, 162, 213], dtype=uint32),
    'queue_log': array([[836, 106,  41,  13,   3,   1,   0,   0,   0,   0],
           [788, 128,  52,  22,   8,   2,   0,   0,   0,   0],
           [623, 186,  96,  54,  24,   9,   5,   3,   0,   0]], dtype=uint32),
    'steps_done': 1000}

    A ill braess graph (simulation ends before completion due to drift).

    >>> sim = VQSimulator(bicycle_graph(), mu=[1, 1, 1, 1], number_events=1000, seed=42, max_queue=10)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([35, 43,  7, 39, 34], dtype=uint32),
    'queue_log': array([[156,  68,  56,  34,  14,   1,   0,   0,   0,   0],
           [306,  19,   3,   1,   0,   0,   0,   0,   0,   0],
           [306,  18,   4,   1,   0,   0,   0,   0,   0,   0],
           [ 98,  67,  35,  25,  26,  30,  16,  11,  10,  11]], dtype=uint32),
    'steps_done': 329}

    A working candy. While candies are not good for greedy policies, the virtual queue is
    designed to deal with it.

    >>> sim = VQSimulator(hyper_paddle(), mu=[1, 1, 1.5, 1, 1.5, 1, 1], number_events=1000, seed=42, max_queue=25)
    >>> sim.run()
    >>> sim.logs # doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([109,  29,  17,  59,  58,  62, 107], dtype=uint32),
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
          dtype=uint32),
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
        self.inners['incid_ptr'] = self.graph.incidence.indptr
        self.inners['incid_ind'] = self.graph.incidence.indices
        self.inners['coinc_ptr'] = self.graph.co_incidence.indptr
        self.inners['coinc_ind'] = self.graph.co_incidence.indices
        self.inners['ready_edges'] = np.zeros(self.graph.m, dtype=bool)
        self.inners['scores'] = np.zeros(self.graph.m, dtype=np.int32)
        self.inners['vq'] = np.zeros(self.graph.m, dtype=np.uint32)
        self.inners['queue_size'] = np.zeros(self.graph.n, dtype=np.uint32)

    def set_core(self):
        """
        Plug in the virtual queue core engine.

        Returns
        -------
        None
        """
        self.core = vq_core


def get_simulator_classes(root=None):
    """
    Parameters
    ----------
    root: :class:`~stochastic_matching.simulator.classes.Simulator`, optional
        Starting class (can be abstract).

    Returns
    -------
    :class:`dict`
        Dictionaries of all subclasses that have a name (as in class attribute `name`).

    Examples
    --------
    >>> get_simulator_classes() # doctest: +NORMALIZE_WHITESPACE
    {'virtual_queue': <class 'stochastic_matching.simulator.classes.VQSimulator'>,
     'random_node': <class 'stochastic_matching.simulator.classes.RandomNode'>,
     'longest_queue': <class 'stochastic_matching.simulator.classes.LongestQueue'>,
     'random_item': <class 'stochastic_matching.simulator.classes.RandomItem'>,
     'fcfm': <class 'stochastic_matching.simulator.classes.FCFM'>}
    """
    if root is None:
        root = Simulator
    result = {c.name: c for c in root.__subclasses__() if c.name}
    for c in root.__subclasses__():
        result.update(get_simulator_classes(c))
    return result


# def display_ccdf(queue_log):
#     """
#     Parameters
#     ----------
#     queue_log: :class:`~numpy.ndarray`
#         Queue matrix (typically the content of 'queue_log'
#         from :attr:`~stochastic_matching.simulator.classes.Simulator.logs`).
#
#     Returns
#     -------
#     :class:`~matplotlib.figure.Figure`
#         A figure of the CCDFs of the queues.
#
#     Examples
#     --------
#
#     >>> from stochastic_matching.graphs.generators import bicycle
#     >>> sim = FCFM(bicycle(), [2, 2.1, 1.1, 1], seed=42)
#     >>> sim.run()
#
#     On IPython, this will draw the CCDF, otherwise a fig object is returned.
#
#     >>> fig = display_ccdf(sim.logs['queue_log'])
#     >>> fig #doctest: +ELLIPSIS
#     <Figure size ...x... with 1 Axes>
#     """
#     # noinspection PyUnresolvedReferences
#     events = np.sum(queue_log[0, :])
#     n, _ = queue_log.shape
#     # noinspection PyUnresolvedReferences
#     ccdf = (events - np.cumsum(np.hstack([np.zeros((n, 1)), queue_log]), axis=1)) / events
#     for i in range(n):
#         plt.semilogy(ccdf[i, ccdf[i, :] > 0], label=f"Node {i}")
#     plt.legend()
#     plt.xlim([0, None])
#     plt.ylim([None, 1])
#     plt.ylabel("CCDF")
#     plt.xlabel("Queue occupancy")
#     return plt.gcf()
