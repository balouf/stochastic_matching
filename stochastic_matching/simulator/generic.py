import numpy as np
from matplotlib import pyplot as plt

from stochastic_matching.display import int_2_str
from stochastic_matching.common import create_prob_alias, graph_neighbors_list


class Simulator:
    """
    Abstract class that describes the generic behavior of matching queues simulator. See sub-classes for examples.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
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

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> sim = sm.FCFM(sm.CycleChain(rates=[2, 2.1, 1.1, 1]), seed=42, number_events=1000, max_queue=8)
    >>> sim
    Simulator of type fcfm.

    Use :meth:`~stochastic_matching.simulator.generic.Simulator.run` to make the simulation.

    >>> sim.run()

    Raw results are stored in `logs`.

    >>> sim.logs #doctest: +NORMALIZE_WHITESPACE
    {'trafic': array([43, 17, 14, 23, 12], dtype=uint64),
    'queue_log': array([[119,  47,  26,  15,  14,   7,   1,   1],
       [189,  25,  13,   3,   0,   0,   0,   0],
       [218,   8,   3,   1,   0,   0,   0,   0],
       [126,  50,  31,  11,   9,   3,   0,   0]], dtype=uint64), 'steps_done': 230}

    Different methods are proposed to provide various indicators.

    >>> sim.compute_average_queues()
    array([1.07826087, 0.26086957, 0.07391304, 0.85217391])

    >>> sim.total_waiting_time()
    0.36535764375876584

    >>> sim.compute_ccdf() #doctest: +NORMALIZE_WHITESPACE
    array([[1.        , 0.4826087 , 0.27826087, 0.16521739, 0.1       ,
        0.03913043, 0.00869565, 0.00434783, 0.        ],
       [1.        , 0.17826087, 0.06956522, 0.01304348, 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [1.        , 0.05217391, 0.0173913 , 0.00434783, 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [1.        , 0.45217391, 0.23478261, 0.1       , 0.05217391,
        0.01304348, 0.        , 0.        , 0.        ]])


    >>> sim.compute_flow()
    array([1.15913043, 0.45826087, 0.3773913 , 0.62      , 0.32347826])

    You can also draw the average or CCDF of the queues.

    >>> fig = sim.show_average_queues()
    >>> fig #doctest: +ELLIPSIS
    <Figure size ...x... with 1 Axes>

    >>> fig = sim.show_average_queues(indices=[0, 3, 2], sort=True, as_time=True)
    >>> fig #doctest: +ELLIPSIS
    <Figure size ...x... with 1 Axes>

    >>> fig = sim.show_ccdf()
    >>> fig #doctest: +ELLIPSIS
    <Figure size ...x... with 1 Axes>

    >>> fig = sim.show_ccdf(indices=[0, 3, 2], sort=True, strict=True)
    >>> fig #doctest: +ELLIPSIS
    <Figure size ...x... with 1 Axes>
    """

    name = None
    """
    Name that can be used to list all non-abstract classes.
    """

    def __init__(self, model, number_events=1000000, seed=None, max_queue=1000, forbidden_edges=None):

        self.forbidden_edges = forbidden_edges

        self.model = model
        self.max_queue = max_queue

        self.generator = None
        self.set_generator(model.rates, number_events, seed)

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
        self.inners = {'neighbors': graph_neighbors_list(self.model),
                       }

    def set_logs(self):
        """
        Populate the monitored variables.

        Returns
        -------

        """
        self.logs = {'trafic': np.zeros(self.model.m, dtype=np.uint64),
                     'queue_log': np.zeros((self.model.n, self.max_queue), dtype=np.uint64),
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
        None
        """
        self.logs['steps_done'] = self.core(**self.generator,
                                            **self.inners, **self.logs)

    def compute_average_queues(self):
        """
        Returns
        -------
        :class:`~numpy.ndarray`
            Average queue sizes.
        """
        return self.logs['queue_log'].dot(np.arange(self.max_queue)) / self.logs['steps_done']

    def total_waiting_time(self):
        """
        Returns
        -------
        :class:`float`
            Average waiting time
        """
        return np.sum(self.compute_average_queues())/np.sum(self.model.rates)

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
        """
        averages = self.compute_average_queues()
        if as_time:
            averages = averages / self.model.rates
        if indices is not None:
            averages = averages[indices]
            names = [int_2_str(self.model, i) for i in indices]
        else:
            names = [int_2_str(self.model, i) for i in range(self.model.n)]
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
        """

        events = self.logs['steps_done']
        n = self.model.n
        # noinspection PyUnresolvedReferences
        return (events - np.cumsum(np.hstack([np.zeros((n, 1)), self.logs['queue_log']]), axis=1)) / events

    def compute_flow(self):
        """
        Normalize the simulated flow.

        Returns
        -------
        None
        """
        # noinspection PyUnresolvedReferences
        tot_mu = np.sum(self.model.rates)
        steps = self.logs['steps_done']
        return self.logs['trafic']*tot_mu/steps

    def show_ccdf(self, indices=None, sort=None, strict=False):
        """
        Parameters
        ----------
        indices: :class:`list`, optional
            Indices of the nodes to display
        sort: :class:`bool`, optional
            If True, order the nodes by decreasing average queue size
        strict: :class:`bool`, default = False
            Draws the curves as a true piece-wise function

        Returns
        -------
        :class:`~matplotlib.figure.Figure`
            A figure of the CCDFs of the queues.
        """
        ccdf = self.compute_ccdf()

        if indices is not None:
            ccdf = ccdf[indices, :]
            names = [int_2_str(self.model, i) for i in indices]
        else:
            names = [int_2_str(self.model, i) for i in range(self.model.n)]
        if sort is True:
            averages = self.compute_average_queues()
            if indices is not None:
                averages = averages[indices]
            ind = np.argsort(-averages)
            ccdf = ccdf[ind, :]
            names = [names[i] for i in ind]
        for i, name in enumerate(names):
            if strict:
                data = ccdf[i, ccdf[i, :] > 0]
                n_d = len(data)
                x = np.zeros(2*n_d-1)
                x[::2] = np.arange(n_d)
                x[1::2] = np.arange(n_d-1)
                y = np.zeros(2 * n_d - 1)
                y[::2] = data
                y[1::2] = data[1:]
                plt.semilogy(x, y, label=name)
            else:
                plt.semilogy(ccdf[i, ccdf[i, :] > 0], label=name)
        plt.legend()
        plt.xlim([0, None])
        plt.ylim([None, 1])
        plt.ylabel("CCDF")
        plt.xlabel("Queue occupancy")
        return plt.gcf()

    def __repr__(self):
        return f"Simulator of type {self.name}."
