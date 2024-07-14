import numpy as np
from numba import njit
from matplotlib import pyplot as plt

from stochastic_matching.simulator.arrivals import Arrivals
from stochastic_matching.simulator.graph import make_jit_graph
from stochastic_matching.simulator.logs import Logs
from stochastic_matching.display import int_2_str
from stochastic_matching.simulator.logs import repr_logs
from stochastic_matching.simulator.metrics import AvgQueues, Regret, Delay, CCDF, Flow


@njit
def core_simulator(logs, arrivals, graph, n_steps, queue_size, selector):
    """
    Generic jitted function for queue-size based policies.

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
    selector: callable
        Jitted function that selects edge (or not) based on graph, queue_size, and arriving node.

    Returns
    -------
    None
    """

    n, max_queue = logs.queue_log.shape

    for age in range(n_steps):

        node = arrivals.draw()
        queue_size[node] += 1
        if queue_size[node] == max_queue:
            return None

        best_edge = selector(graph=graph, queue_size=queue_size, node=node)

        if best_edge > -1:
            queue_size[graph.nodes(best_edge)] -= 1

        logs.update(queue_size=queue_size, node=node, edge=best_edge)


class Simulator:
    """
    Abstract class that describes the generic behavior of matching simulators.
    See subclasses for detailed examples.

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
        Inner variables. Default to arrivals, graphs, queue_size, n_steps. Subclasses can add other variables.
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

    Raw results are stored in `logs`. A `plog` property gives a pretty print of the logs.

    >>> sim.plogs #doctest: +NORMALIZE_WHITESPACE
    Arrivals: [67 80 43 39]
    Traffic: [43 17 14 23 12]
    Queues: [[118  47  26  15  14   7   1   1]
     [188  25  13   3   0   0   0   0]
     [217   8   3   1   0   0   0   0]
     [125  50  31  11   9   3   0   0]]
    Steps done: 229

    Different methods are proposed to provide various indicators.

    >>> sim.avg_queues
    array([1.08296943, 0.26200873, 0.07423581, 0.8558952 ])

    >>> sim.delay
    0.3669530919847866

    >>> sim.ccdf #doctest: +NORMALIZE_WHITESPACE
    array([[1.        , 0.48471616, 0.27947598, 0.16593886, 0.10043668,
            0.03930131, 0.00873362, 0.00436681, 0.        ],
           [1.        , 0.1790393 , 0.069869  , 0.01310044, 0.        ,
            0.        , 0.        , 0.        , 0.        ],
           [1.        , 0.05240175, 0.01746725, 0.00436681, 0.        ,
            0.        , 0.        , 0.        , 0.        ],
           [1.        , 0.45414847, 0.23580786, 0.10043668, 0.05240175,
            0.01310044, 0.        , 0.        , 0.        ]])
    >>> sim.flow
    array([1.16419214, 0.46026201, 0.3790393 , 0.62270742, 0.32489083])

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

        self.logs = Logs(n=self.model.n, m=self.model.m, max_queue=self.max_queue)

    @property
    def plogs(self):
        """
        Print logs.

        Returns
        -------
        None
        """
        repr_logs(self.logs)

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
        self.logs = Logs(n=self.model.n, m=self.model.m, max_queue=self.max_queue)

    def run(self):
        """
        Run simulation.
        Results are stored in the attribute :attr:`~stochastic_matching.simulator.simulator.Simulator.logs`.

        Returns
        -------
        None
        """
        core_simulator(logs=self.logs, **self.internal)

    @property
    def avg_queues(self):
        """
        :class:`~numpy.ndarray`
            Average queue sizes.
        """
        return AvgQueues.get(self)

    @property
    def delay(self):
        """
        :class:`float`
            Average waiting time
        """
        return Delay.get(self)

    @property
    def ccdf(self):
        """
        :class:`~numpy.ndarray`
            CCDFs of the queues.
        """
        return CCDF.get(self)

    @property
    def flow(self):
        """
        Normalize the simulated flow.

        Returns
        -------
        :class:`~numpy.ndarray`
            Flow on edges.
        """
        return Flow.get(self)

    @property
    def regret(self):
        return Regret.get(self)

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
        averages = self.avg_queues
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
        ccdf = self.ccdf

        if indices is not None:
            ccdf = ccdf[indices, :]
            names = [int_2_str(self.model, i) for i in indices]
        else:
            names = [int_2_str(self.model, i) for i in range(self.model.n)]
        if sort is True:
            averages = self.avg_queues
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
