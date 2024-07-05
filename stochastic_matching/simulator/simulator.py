import numpy as np
from numba import njit
from matplotlib import pyplot as plt

from stochastic_matching.simulator.arrivals import Arrivals
from stochastic_matching.simulator.graph import make_jit_graph
from stochastic_matching.simulator.logs import Logs
from stochastic_matching.display import int_2_str


@njit
def core_simulator(arrivals, graph, n_steps, queue_size, selector,
                   traffic, queue_log, steps_done):
    """
    Generic jitted function for queue-size based policies.

    Parameters
    ----------
    arrivals: :class:`~stochastic_matching.simulator.arrivals.Arrivals`
        Item arrival process.
    graph: :class:`~stochastic_matching.simulator.graph.JitHyperGraph`
        Model graph.
    n_steps: :class:`int`
        Number of arrivals to process.
    queue_size: :class:`~numpy.ndarray`
        Number of waiting items of each class.
    selector: callable
        Jitted function that selects edge (or not) based on graph, queue_size, and arriving node.
    traffic: :class:`~numpy.ndarray`
        Monitor traffic on edges.
    queue_log: :class:`~numpy.ndarray`
        Monitor queue sizes.
    steps_done: :class:`int`
        Number of arrivals processed so far.

    Returns
    -------
    :class:`int`
        Total number of steps performed, possibly including previous runs.
    """

    n, max_queue = queue_log.shape

    for age in range(n_steps):
        for j in range(n):
            queue_log[j, queue_size[j]] += 1

        # Draw an arrival
        node = arrivals.draw()
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return steps_done + age + 1

        best_edge = selector(graph=graph, queue_size=queue_size, node=node)

        if best_edge > -1:
            traffic[best_edge] += 1
            queue_size[graph.nodes(best_edge)] -= 1
    return steps_done + age + 1


class Simulator:
    """
    Abstract class that describes the generic behavior of matching queues simulator. See sub-classes for examples.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    n_steps: :class:`int`, optional
        Number of arrivals to simulate.
    seed: :class:`int`, optional
        Seed of the random generator
    max_queue: :class:`int`
        Max queue size. Necessary for speed and detection of unstability.
        For stable systems very close to the unstability
        border, the max_queue may be reached.

    Attributes
    ----------
    internal: :class:`dict`
        Inner variables. Default to arrivals, graphs, queue_size, n_steps. Sub-classes can add other variables.
    logs: :class:`~stochastic_matching.simulator.logs.Logs`
        Monitored variables (traffic on edges, queue size distribution, number of steps achieved).

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> sim = sm.FCFM(sm.CycleChain(rates=[2, 2.1, 1.1, 1]), seed=42, n_steps=1000, max_queue=8)
    >>> sim
    Simulator of type fcfm.

    Use :meth:`~stochastic_matching.simulator.simulator.Simulator.run` to make the simulation.

    >>> sim.run()

    Raw results are stored in `logs`.

    >>> sim.logs #doctest: +NORMALIZE_WHITESPACE
    Traffic: [43 17 14 23 12]
    Queues: [[119  47  26  15  14   7   1   1]
     [189  25  13   3   0   0   0   0]
     [218   8   3   1   0   0   0   0]
     [126  50  31  11   9   3   0   0]]
    Steps done: 230

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

    def __init__(self, model, n_steps=1000000, seed=None, max_queue=1000):

        self.model = model
        self.max_queue = max_queue
        self.n_steps = n_steps
        self.seed = seed

        self.internal = None
        self.set_internal()

        self.logs = Logs(self)

    def set_internal(self):
        """
        Populate the internal state.

        Returns
        -------
        None
        """
        self.internal = {'arrivals': Arrivals(mu=self.model.rates, seed=self.seed),
                         'graph': make_jit_graph(self.model),
                         'n_steps': self.n_steps,
                         'queue_size': np.zeros(self.model.n, dtype=int)
                         }

    def reset(self):
        """
        Reset internal state and monitored variables.

        Returns
        -------
        None
        """
        self.set_internal()
        self.logs = Logs(self)

    def run(self):
        """
        Run simulation.
        Results are stored in the attribute :attr:`~stochastic_matching.simulator.simulator.Simulator.logs`.

        Returns
        -------
        None
        """
        self.logs.steps_done = core_simulator(**self.internal, **self.logs.asdict())

    def compute_average_queues(self):
        """
        Returns
        -------
        :class:`~numpy.ndarray`
            Average queue sizes.
        """
        return self.logs.queue_log.dot(np.arange(self.max_queue)) / self.logs.steps_done

    def total_waiting_time(self):
        """
        Returns
        -------
        :class:`float`
            Average waiting time
        """
        return np.sum(self.compute_average_queues()) / np.sum(self.model.rates)

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
        events = self.logs.steps_done
        n = self.model.n
        # noinspection PyUnresolvedReferences
        return (events - np.cumsum(np.hstack([np.zeros((n, 1)), self.logs.queue_log]), axis=1)) / events

    def compute_flow(self):
        """
        Normalize the simulated flow.

        Returns
        -------
        :class:`~numpy.ndarray`
            Flow on edges.
        """
        # noinspection PyUnresolvedReferences
        tot_mu = np.sum(self.model.rates)
        steps = self.logs.steps_done
        return self.logs.traffic * tot_mu / steps

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
                x = np.zeros(2 * n_d - 1)
                x[::2] = np.arange(n_d)
                x[1::2] = np.arange(n_d - 1)
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
