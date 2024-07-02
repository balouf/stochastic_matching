import numpy as np
from matplotlib import pyplot as plt

from stochastic_matching.newsimulator.arrivals import Arrivals
from stochastic_matching.newsimulator.graph import make_jit_graph
from stochastic_matching.display import int_2_str


class NewSimulator:
    name = None
    """
    Name that can be used to list all non-abstract classes.
    """

    def __init__(self, model, n_steps=1000000, seed=None, max_queue=1000):

        self.model = model
        self.max_queue = max_queue
        self.n_steps = n_steps
        self.seed = seed

        self.state = None
        self.set_state()

        self.logs = None
        self.set_logs()

        self.core = None

    def set_state(self):
        """
        Populate the internal state.

        Returns
        -------
        None
        """
        self.state = {'arrivals': Arrivals(mu=self.model.rates, seed=self.seed),
                      'graph': make_jit_graph(self.model),
                      'n_steps': self.n_steps
                      }

    def set_logs(self):
        """
        Populate the monitored variables.

        Returns
        -------
        None
        """
        self.logs = {'trafic': np.zeros(self.model.m, dtype=int),
                     'queue_log': np.zeros((self.model.n, self.max_queue), dtype=int),
                     'steps_done': 0}

    def reset(self):
        """
        Reset internal state and monitored variables.

        Returns
        -------
        None
        """
        self.set_state()
        self.set_logs()

    def run(self):
        """
        Run simulation.
        Results are stored in the attribute :attr:`~stochastic_matching.simulator.simulator.Simulator.logs`.

        Returns
        -------
        None
        """
        self.logs['steps_done'] = self.core(**self.state, **self.logs)

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
        return self.logs['trafic'] * tot_mu / steps

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
