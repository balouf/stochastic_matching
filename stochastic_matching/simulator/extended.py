import numpy as np

from stochastic_matching.simulator.simulator import Simulator


class ExtendedSimulator(Simulator):
    """
    This extended (abstract) class of :class:`~stochastic_matching.simulator.simulator.Simulator` accepts
    additional parameters designed to reach vertices of the stability polytope.

    How these parameters are exactly used is determined by its sub-classes.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    rewards: :class:`~numpy.ndarray` or :class:`list`, optional
        Rewards associated to edges.
    beta: :class:`float`, default=.01
        Reward factor: each edge is given a default score 1/beta (small beta means higher reward impact).
    forbidden_edges: :class:`list` or :class:`~numpy.ndarray`, or :class:`bool`, optional
        Edges that should not be used. If set to True, the list of forbidden edges is computed from the rewards.
    k: :class:`int`, optional
        Limit on queue size to apply edge interdiction (enforce stability on injective-only vertices).
    kwargs: :class:`dict`
        Keyword parameters of :class:`~stochastic_matching.simulator.simulator.Simulator`
    """
    def __init__(self, model, rewards=None, beta=.01,
                 forbidden_edges=None, k=None,
                 **kwargs):
        self.rewards = np.array(rewards) if rewards is not None else None
        self.beta = beta
        if forbidden_edges is True:
            flow = model.optimize_rates(self.rewards)
            forbidden_edges = [i for i in range(model.m) if flow[i] == 0]
            if len(forbidden_edges) == 0:
                forbidden_edges = None
            self.rewards = None
        self.forbidden_edges = forbidden_edges
        self.k = k
        super().__init__(model, **kwargs)

    def set_internal(self):
        super().set_internal()
        if self.rewards is None:
            scores = np.zeros(self.model.m, dtype=int)
        else:
            scores = self.rewards / self.beta
        self.internal['scores'] = scores
        self.internal['forbidden_edges'] = self.forbidden_edges
        self.internal['k'] = self.k
