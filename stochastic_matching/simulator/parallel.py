from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy


class VariableParameter:
    """
    Allows to vary one single parameter of simulation for experiments.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to simulate.
    name: :class:`str`
        Name of the experiments.
    key: :class:`str`
        Variable parameter.
    values: iterable
        Values of the parameter
    kwargs: :class:`dict`
        All other keyword parameters will be passed to simulation (including the choice of `simulator`).
    """
    def __init__(self, model, name, key, values, **kwargs):
        self.name = name
        self.key = key
        self.values = values
        self.model = deepcopy(model)
        self.model.simulator = None
        self.kwargs = kwargs

    def __iter__(self):
        for v in self.values:
            params = {**self.kwargs, self.key: v}
            yield self.name, self.key, params, self.model


def build_metric_computer(metric_extractor=None):
    """
    Parameters
    ----------
    metric_extractor: callable, optional
        The metric extractor must have a (params, model) signature and return a dictionary of metrics.

    Returns
    -------
    callable
        A function with signature (name, key, params, model).
    """
    if metric_extractor is None:
        metric_extractor = regret_delay

    def compute(name, key, params, model):
        model.run(**params)
        return {'name': name, key: params[key], **metric_extractor(params, model)}

    return compute


def regret_delay(params, model):
    """
    Parameters
    ----------
    params: :class:`dict`
        Parameters of the simulation
    model: :class:`~stochastic_matching.model.Model`
        Model simulated.

    Returns
    -------
    :class:`dict`
        Regret (implies that params has a `rewards` key) and delay.
    """
    simu = model.simulator
    original_rates = model.rates
    model.rates = model.incidence @ simu.compute_flow()
    rewards = params['rewards']
    best_flow = model.optimize_rates(rewards)
    model.rates = original_rates
    flow = model.simulator.compute_flow()
    regret = rewards @ (best_flow - flow)
    delay = sum(model.simulator.compute_average_queues())
    return {'regret': regret, 'delay': delay}


def aggregate(results):
    """
    Parameters
    ----------
    results: :class:`list`
        Computed results. Each entry is a dictionary associated to a given experiments.
    Returns
    -------
    :class:`dict`
        All results are gathered by experiment name, then by varying input / metric.
    """
    res = dict()
    for r in results:
        name = r.pop('name')
        if name not in res:
            res[name] = {k: [] for k in r}
        for k, v in r.items():
            res[name][k].append(v)
    return res


def evaluate(xps, n_jobs=-1, metric_extractor=None):
    """
    Parameters
    ----------
    xps: :class:`~stochastic_matching.simulator.parallel.VariableParameter` or :class:`list`
        Experiment(s)
    n_jobs: :class:`int`, default=-1
        Number of workers, joblib-style (cf https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html).
    metric_extractor: callable, optional
        The metric extractor must have a (params, model) signature and return a dictionary of metrics.

    Returns
    -------
    :class:`dict`
        Result of the experiment(s).

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> import numpy as np
    >>> diamond = sm.CycleChain()
    >>> base = {'model': diamond, 'n_steps': 1000, 'seed': 42, 'rewards': [1, 2.9, 1, -1, 1]}
    >>> xp1 = VariableParameter(name='e-filtering', simulator='e_filtering',
    ...                         key='epsilon', values=[.01, .1, 1], **base)
    >>> xp2 = VariableParameter(name='k-filtering', simulator='longest', forbidden_edges=True,
    ...                         key='k', values=[0, 10, 100], **base)
    >>> xp3 = VariableParameter(name='egpd', simulator='virtual_queue',
    ...                         key='beta', values=[.01, .1, 1], **base)
    >>> res = evaluate(xp1)
    >>>
    >>> for k, v in res['e-filtering'].items():
    ...     print(f"{k}: {np.array(v)}")
    epsilon: [0.01 0.1  1.  ]
    regret: [0.002 0.017 0.103]
    delay: [10.534  6.948  1.948]
    >>> res = evaluate([xp2, xp3], n_jobs=2)
    >>> for name, r in res.items():
    ...     print(name)
    ...     for k, v in r.items():
    ...         print(f"{k}: {np.array(v)}")
    k-filtering
    k: [  0  10 100]
    regret: [ 8.8000000e-02  2.0000000e-03 -8.8817842e-16]
    delay: [ 1.634  7.342 13.524]
    egpd
    beta: [0.01 0.1  1.  ]
    regret: [0.003 0.043 0.076]
    delay: [91.922 10.118  1.886]
    """
    if isinstance(xps, list):
        jobs = [x for xp in xps for x in xp]
    else:
        jobs = [x for x in xps]
    compute = build_metric_computer(metric_extractor)
    res = Parallel(n_jobs=n_jobs)(delayed(compute)(*args) for args in tqdm(jobs))
    return aggregate(res)
