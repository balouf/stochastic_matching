import numpy as np
from stochastic_matching.spectral import Spectral
from stochastic_matching.simulator.classes import Simulator, get_simulator_classes
from inspect import isclass


class MQ:
    """
    The MQ class is the main point of entry to play with matching queues.

    Parameters
    ----------
    graph: :class:`~stochastic_matching.graphs.classes.SimpleGraph` or :class:`~stochastic_matching.graph.classes.HyperGraph`, optional
        Graph to analyze.
    mu: :class:`~numpy.ndarray` or :class:`list`, optional
        Arrival rates.
    tol: :class:`float`, optional
        Values of absolute value lower than `tol` are set to 0.
    """
    def __init__(self, graph=None, mu=None, tol=1e-9):
        self.spectral = Spectral(tol=tol)
        self.graph = graph
        self.mu = mu
        self.flow = None
        self.fit(graph=graph, mu=mu)
        self.simulator = None
        self.simulation_flow = None

    def maximin_flow(self):
        """
        Maximizes the minimal flow over all edges.

        Returns
        -------
        :class:`~numpy.ndarray`
            Optimized flow.

        Examples
        --------

        Consider the following Braess example.

        >>> from stochastic_matching.graphs.generators import bicycle_graph
        >>> braess = bicycle_graph()
        >>> mq = MQ(braess, [1, 3, 2, 2])

        Let us see the base flow.

        >>> mq.spectral.base_flow
        array([0.75, 0.25, 1.  , 1.25, 0.75])

        Lowest edge load is 0.25. Let us find the maximin solution.

        >>> mq.maximin_flow()
        array([0.5, 0.5, 1. , 1.5, 0.5])

        Another similar example.

        >>> braess = bicycle_graph(right_cycle=4)
        >>> mq = MQ(braess, [7, 4, 4, 2, 2])

        Let us see the base flow. It has a negative value!

        >>> mq.spectral.base_flow
        array([ 3.5  ,  3.5  , -0.125,  0.625,  0.625,  1.375])

        However, we still have a positive solution.

        >>> mq.maximin_flow()
        array([3.5       , 3.5       , 0.25000001, 0.24999999, 0.24999999,
               1.75000001])

        Note that for graphs with trivial kernel, the solution is unique and the optimizer will directly return it.

        >>> from stochastic_matching.graphs.generators import hyper_paddle
        >>> candy = hyper_paddle()
        >>> mq = MQ(candy, [1, 1, 2, 1, 2, 1, 1])

        Let us *maximin* the flow.

        >>> mq.maximin_flow()
        array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. ])
        """
        self.flow = self.spectral.maximin_flow()
        return self.flow

    def incompressible_flow(self):
        """
        Finds the minimal flow that must pass through each edge. This is currently done in a *brute force* way
        by minimizing every edges.

        Returns
        -------
        :class:`~numpy.ndarray`
            Unavoidable flow.

        Examples
        --------

        Consider the following Braess example.

        >>> from stochastic_matching.graphs.generators import bicycle_graph
        >>> braess = bicycle_graph()
        >>> mq = MQ(braess, [1, 3, 2, 2])

        Let us see the base flow.

        >>> mq.spectral.base_flow
        array([0.75, 0.25, 1.  , 1.25, 0.75])

        What is the part that mus always be there?

        >>> mq.incompressible_flow()
        array([0., 0., 1., 1., 0.])

        Another similar example.

        >>> braess = bicycle_graph(right_cycle=4)
        >>> mq = MQ(braess, [7, 4, 4, 2, 2])

        Let us see the base flow. It has a negative value!

        >>> mq.spectral.base_flow
        array([ 3.5  ,  3.5  , -0.125,  0.625,  0.625,  1.375])

        What is necessary in all positive solutions?

        >>> mq.incompressible_flow()
        array([3.5, 3.5, 0. , 0. , 0. , 1.5])

        Note that for graphs with trivial kernel, the solution is unique and the optimizer will directly return it.

        >>> from stochastic_matching.graphs.generators import hyper_paddle
        >>> candy = hyper_paddle()
        >>> mq = MQ(candy, [1, 1, 2, 1, 2, 1, 1])

        Let us *maximin* the flow.

        >>> mq.incompressible_flow()
        array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. ])
        """
        self.flow = self.spectral.incompressible_flow()
        return self.flow

    def optimize_edge(self, edge, sign=1):
        """
        Tries to find a positive solution that minimizes/maximizes a given edge.

        Parameters
        ----------
        edge: :class:`int`
            Edge to optimize.
        sign: :class:`int`
            Use 1 to maximize, -1 to minimize.

        Returns
        -------
        :class:`~numpy.ndarray`
            Optimized flow.

        Examples
        --------

        Consider the following Braess example.

        >>> from stochastic_matching.graphs.generators import bicycle_graph
        >>> braess = bicycle_graph()
        >>> mq = MQ(braess, [1, 2, 2, 1])

        All border edges can vary between 0 and 1. Let us maximize the first edge.

        >>> mq.optimize_edge(0, 1)
        array([1., 0., 1., 0., 1.])

        Let us minimize the first edge.

        >>> mq.optimize_edge(0, -1)
        array([0., 1., 1., 1., 0.])

        Note that for graphs with trivial kernel, the solution is unique and the optimizer will directly return it.

        >>> from stochastic_matching.graphs.generators import hyper_paddle
        >>> candy = hyper_paddle()
        >>> mq = MQ(candy, [1, 1, 2, 1, 2, 1, 1])

        Let us *maximize* the first edge.

        >>> mq.optimize_edge(0, 1)
        array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. ])

        Let us *minimize* the first edge.

        >>> mq.optimize_edge(0, -1)
        array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. ])
        """
        self.flow = self.spectral.optimize_edge(edge, sign=sign)
        return self.flow

    def show_flow(self, flow=None, check=True, tol=1e-2, options=None):
        """
        Display flow.

        Parameters
        ----------
        flow: :class:`~numpy.ndarray`, optional
            Flow to display. If no flow is specified, the last computed flow is used.
        check: :class:`bool`, optional
            If True, validity of flow will be displayed: nodes that do not check the conservation law will be red,
            negative edges will be red, null edges will be orange.
        tol: :class:`float`, optional
            Relative tolerance for the checking of conservation law.
        options: :class:`dict`
            Options to pass to the vis engine.

        Returns
        -------
        :class:`~IPython.display.HTML`
            Displayed graph.

        Examples
        --------

        >>> from stochastic_matching import tadpole_graph, fan, bicycle_graph

        Example with a red (negative) edge.

        >>> mq = MQ(bicycle_graph(right_cycle=4), mu=[7, 4, 4, 2, 2])
        >>> mq.spectral.base_flow
        array([ 3.5  ,  3.5  , -0.125,  0.625,  0.625,  1.375])
        >>> mq.show_flow()
        <IPython.core.display.HTML object>

        Example with an orange (null) edge.

        >>> mq = MQ(tadpole_graph())
        >>> mq.spectral.base_flow
        array([1., 0., 0., 1.])
        >>> mq.show_flow()
        <IPython.core.display.HTML object>

        Example with red nodes (broken conservation law).

        >>> mq = MQ(tadpole_graph(m=4))

        If conservation law holds, the following should be made of 1's.

        >>> mq.graph.incidence.dot(mq.spectral.base_flow)
        array([0.8, 1.2, 0.8, 1.2, 0.8])
        >>> mq.show_flow()
        <IPython.core.display.HTML object>

        Example on a hypergraph.

        >>> mq = MQ(graph=fan(hyperedges=2))
        >>> mq.maximin_flow()
        array([0.25, 0.5 , 0.5 , 0.25, 0.5 , 0.5 , 0.25, 0.5 , 0.5 , 0.25, 0.25])
        >>> mq.show_flow()
        <IPython.core.display.HTML object>
        """
        if flow is None:
            flow = self.flow
        nodes_dict = [{'label': f"{self.mu[i]:.2f}"} for i in range(self.graph.n)]
        edges_dict = [{'label': f"{flow[j]:.2f}"} for j in range(self.graph.m)]
        if check:
            mueff = self.graph.incidence.dot(flow)
            for i in range(self.graph.n):
                if np.abs(self.mu[i] - mueff[i]) / self.mu[i] > tol:
                    nodes_dict[i]['color'] = 'red'
            for j in range(self.graph.m):
                if flow[j] < 0:
                    edges_dict[j]['color'] = 'red'
                elif flow[j] == 0:
                    edges_dict[j]['color'] = 'orange'
        self.graph.show(options=options, nodes_dict=nodes_dict, edges_dict=edges_dict, )

    def fit(self, graph=None, mu=None):
        """
        Compute internal attributes for graph and/or rate.
        If `graph` is provided without `mu`, uniform rate is assumed.

        Parameters
        ----------
        graph: :class:`~stochastic_matching.graphs.classes.GenericGraph`, optional
            Graph to analyze.
        mu: :class:`~numpy.ndarray` or :class:`list`, optional
            Arrival rates.

        Returns
        -------
        None
        """
        if graph is not None:
            self.graph = graph
            if mu is None:
                mu = np.ones(graph.n)
        self.mu = mu
        self.spectral.fit(graph, mu)
        if self.spectral.base_flow is not None:
            self.flow = self.spectral.base_flow

    def set_simulator(self, simulator,
                      number_events=1000000, seed=None, max_queue=1000):
        """
        Instantiate simulator.

        Parameters
        ----------
        simulator: :class:`str` or :class:`~stochastic_matching.simulator.classes.Simulator`
            Type of simulator to instantiate.
        number_events: :class:`int`, optional
            Number of arrivals to simulate.
        seed: :class:`int`, optional
            Seed of the random generator
        max_queue: :class:`int`
            Max queue size. Necessary for speed and detection of unstability.

        Returns
        -------
        None

        Examples
        --------
        >>> from stochastic_matching import bicycle_graph
        >>> mq = MQ(bicycle_graph())
        >>> mq.set_simulator('something')  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: something is not a known simulator name. Known names: virtual_queue, random_node,
        longest_queue, random_item, fcfm.

        >>> mq.set_simulator('fcfm')
        >>> mq.simulator.inners.keys()
        dict_keys(['neighbors', 'queue_start', 'queue_end', 'items'])

        >>> from stochastic_matching.simulator.classes import RandomNode
        >>> mq.set_simulator(RandomNode)
        >>> mq.simulator.inners.keys()
        dict_keys(['neighbors', 'queue_size'])

        >>> mq.set_simulator(RandomNode(bicycle_graph(), [1, 1, 1, 1]))
        Traceback (most recent call last):
        ...
        TypeError: simulator must be string or Simulator class (not instance).
        """
        if isinstance(simulator, str):
            simu_dict = get_simulator_classes()
            if simulator in simu_dict:
                self.simulator = simu_dict[simulator](self.graph, self.mu,
                                                      number_events, seed, max_queue)
            else:
                raise ValueError(f"{simulator} is not a known simulator name. "
                                 f"Known names: {', '.join(simu_dict.keys())}.")
        elif isclass(simulator) and issubclass(simulator, Simulator):
            self.simulator = simulator(self.graph, self.mu, number_events, seed, max_queue)
        else:
            raise TypeError("simulator must be string or Simulator class (not instance).")

    def get_simulation_flow(self):
        """
        Normalize the simulated flow.

        Returns
        -------
        None
        """
        # noinspection PyUnresolvedReferences
        tot_mu = np.sum(self.mu)
        steps = self.simulator.logs['steps_done']
        self.simulation_flow = self.simulator.logs['trafic']*tot_mu/steps

    def run(self, simulator, number_events=1000000, seed=None, max_queue=1000):
        """
        All-in-one instantiate and run simulation.

        Parameters
        ----------
        simulator: :class:`str` or :class:`~stochastic_matching.simulator.classes.Simulator`
            Type of simulator to instantiate.
        number_events: :class:`int`, optional
            Number of arrivals to simulate.
        seed: :class:`int`, optional
            Seed of the random generator
        max_queue: :class:`int`
            Max queue size. Necessary for speed and detection of unstability.

        Returns
        -------
        bool
            Success of simulation.

        Examples
        --------

        Let start with a working triangle and a greedy simulator.

        >>> from stochastic_matching import tadpole_graph, bicycle_graph, hyper_paddle
        >>> mq = MQ(tadpole_graph(n=0), mu=[3, 4, 5])
        >>> mq.spectral.base_flow
        array([1., 2., 3.])
        >>> mq.run('random_node', seed=42, number_events=20000)
        True
        >>> mq.flow
        array([1.044 , 2.0352, 2.9202])

        A ill braess graph (simulation ends before completion due to drift).

        Note that the drift is slow, so if the number of steps is low the simulation may complete without overflowing.

        >>> mq.fit(bicycle_graph(), mu=[1, 1, 1, 1])
        >>> mq.spectral.base_flow
        array([0.5, 0.5, 0. , 0.5, 0.5])

        >>> mq.run('longest_queue', seed=42, number_events=20000)
        True
        >>> mq.flow
        array([0.501 , 0.4914, 0.0018, 0.478 , 0.5014])

        A working candy. While candies are not good for greedy policies, the virtual queue is
        designed to deal with it.

        >>> mq.fit(hyper_paddle(), mu=[1, 1, 1.1, 1, 1.1, 1, 1])
        >>> mq.spectral.base_flow
        array([0.95, 0.05, 0.05, 0.05, 0.05, 0.95, 1.  ])

        The above states that the target flow for the hyperedge of the candy (last entry) is 1.

        >>> mq.run('longest_queue', seed=42, number_events=20000)
        False
        >>> mq.simulator.logs['steps_done']
        10459
        >>> mq.flow  # doctest: +NORMALIZE_WHITESPACE
        array([0.64227938, 0.37586767, 0.38757051, 0.40753418, 0.40891099,
           0.59202601, 0.2939478 ])

        A greedy simulator performs poorly on the hyperedge.

        >>> mq.run('virtual_queue', seed=42, number_events=20000)
        True
        >>> mq.flow  # doctest: +NORMALIZE_WHITESPACE
        array([0.96048, 0.04104, 0.04428, 0.06084, 0.06084, 0.94464, 0.9846 ])

        The virtual queue simulator manages to cope with the target flow on the hyperedge.
        """
        self.set_simulator(simulator, number_events, seed, max_queue)
        self.simulator.run()
        self.get_simulation_flow()
        self.flow = self.simulation_flow
        return number_events == self.simulator.logs['steps_done']
