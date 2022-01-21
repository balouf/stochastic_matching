import numpy as np
from stochastic_matching.simulator.classes import Simulator
from stochastic_matching.common import get_classes
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
    def __init__(self, model):
        self.model = model
        self.flow = None
        self.simulator = None
        self.simulation_flow = None

    def maximin_flow(self):
        """
        Maximizes the minimal flow over all edges.

        Returns
        -------
        :class:`~numpy.ndarray`
            Optimized flow.

        """
        self.flow = self.model.maximin
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

        >>> from stochastic_matching.graphs import CycleChain
        >>> diamond = CycleChain(rates=[1, 3, 2, 2])
        >>> mq = MQ(diamond)

        Let us see the base flow.

        >>> mq.model.base_flow
        array([0.75, 0.25, 1.  , 1.25, 0.75])

        What is the part that mus always be there?

        >>> mq.incompressible_flow()
        array([0., 0., 1., 1., 0.])

        Another similar example.

        >>> from stochastic_matching.graphs import Cycle, concatenate
        >>> house = concatenate([Cycle(), Cycle(4)], overlap=2, rates=[7, 4, 4, 2, 2])
        >>> mq = MQ(house)

        Let us see the base flow. It has a negative value!

        >>> mq.model.base_flow
        array([ 3.5  ,  3.5  , -0.125,  0.625,  0.625,  1.375])

        What is necessary in all positive solutions?

        >>> mq.incompressible_flow()
        array([3.5, 3.5, 0. , 0. , 0. , 1.5])

        Note that for graphs with trivial kernel, the solution is unique and the optimizer will directly return it.

        >>> from stochastic_matching.graphs import HyperPaddle
        >>> candy = HyperPaddle(rates=[1, 1, 2, 1, 2, 1, 1])
        >>> mq = MQ(candy)

        Let us *maximin* the flow.

        >>> mq.incompressible_flow()
        array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. ])
        """
        self.flow = self.model.incompressible_flow()
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

        """
        self.flow = self.model.optimize_edge(edge, sign=sign)
        return self.flow

    def show_flow(self, **kwargs):
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

        >>> from stochastic_matching.graphs import concatenate, Cycle, CycleChain, Fan, Tadpole

        Example with a red (negative) edge.

        >>> mq = MQ(concatenate([Cycle(), Cycle(4)], overlap=2, rates=[7, 4, 4, 2, 2]))
        >>> mq.model.base_flow
        array([ 3.5  ,  3.5  , -0.125,  0.625,  0.625,  1.375])
        >>> mq.show_flow()
        <IPython.core.display.HTML object>

        Example with an orange (null) edge.

        >>> mq = MQ(Tadpole(rates='uniform'))
        >>> mq.model.base_flow
        array([1., 0., 0., 1.])
        >>> mq.show_flow()
        <IPython.core.display.HTML object>

        Example with red nodes (broken conservation law).

        >>> mq = MQ(Tadpole(m=4, rates='uniform'))

        If conservation law holds, the following should be made of 1's.

        >>> mq.model.incidence @ mq.model.base_flow
        array([0.8, 1.2, 0.8, 1.2, 0.8])
        >>> mq.show_flow()
        <IPython.core.display.HTML object>

        Example on a hypergraph.

        >>> mq = MQ(Fan(hyperedges=2, rates='uniform'))
        >>> mq.maximin_flow()
        array([0.25, 0.5 , 0.5 , 0.25, 0.5 , 0.5 , 0.25, 0.5 , 0.5 , 0.25, 0.25])
        >>> mq.show_flow()
        <IPython.core.display.HTML object>
        """
        self.model.show_flow(**kwargs)

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
        >>> from stochastic_matching.graphs import CycleChain
        >>> mq = MQ(CycleChain())
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

        >>> mq.set_simulator(RandomNode(CycleChain(rates='uniform')))
        Traceback (most recent call last):
        ...
        TypeError: simulator must be string or Simulator class (not instance).
        """
        if isinstance(simulator, str):
            simu_dict = get_classes(Simulator)
            if simulator in simu_dict:
                self.simulator = simu_dict[simulator](self.model, number_events, seed, max_queue)
            else:
                raise ValueError(f"{simulator} is not a known simulator name. "
                                 f"Known names: {', '.join(simu_dict.keys())}.")
        elif isclass(simulator) and issubclass(simulator, Simulator):
            self.simulator = simulator(self.model, number_events, seed, max_queue)
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
        tot_mu = np.sum(self.model.rates)
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

        >>> from stochastic_matching.graphs import Tadpole, CycleChain, HyperPaddle, Cycle
        >>> mq = MQ(Cycle(rates=[3, 4, 5]))
        >>> mq.model.base_flow
        array([1., 2., 3.])
        >>> mq.run('random_node', seed=42, number_events=20000)
        True
        >>> mq.flow
        array([1.044 , 2.0352, 2.9202])

        A ill braess graph (simulation ends before completion due to drift).

        Note that the drift is slow, so if the number of steps is low the simulation may complete without overflowing.

        >>> mq.model = CycleChain(rates='uniform')
        >>> mq.model.base_flow
        array([0.5, 0.5, 0. , 0.5, 0.5])

        >>> mq.run('longest_queue', seed=42, number_events=20000)
        True
        >>> mq.flow
        array([0.501 , 0.4914, 0.0018, 0.478 , 0.5014])

        A working candy. While candies are not good for greedy policies, the virtual queue is
        designed to deal with it.

        >>> mq.model = HyperPaddle(rates=[1, 1, 1.1, 1, 1.1, 1, 1])
        >>> mq.model.base_flow
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
        self.set_simulator(simulator, number_events=number_events, seed=seed, max_queue=max_queue)
        self.simulator.run()
        self.get_simulation_flow()
        self.flow = self.simulation_flow
        return number_events == self.simulator.logs['steps_done']
