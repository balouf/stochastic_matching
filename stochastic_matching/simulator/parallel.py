from tqdm import tqdm
from pathlib import Path
import gzip
import dill as pickle
import functools


def do_nothing(x):
    """

    Parameters
    ----------
    x: object
        Something

    Returns
    -------
    object
        Same object
    """
    return x


class Iterator:
    """
    Provides an easy way to make a parameter vary.

    Parameters
    ----------
    parameter: :class:`str`
        Name of the argument to vary.
    values: iterable
        Values that the argument can take.
    name: :class:`str`, optional
        *Display* name of the parameter
    process: callable, optional
        If you want to transform the value used, use this.


    Returns
    -------
    kwarg: :class:`dict`
        Keyword argument to use.
    log: :class:
        What you want to remind. By default, identical to kwarg.


    Examples
    --------

    Imagine one wants a parameter x2 to vary amongst squares of integers.

    >>> iterator = Iterator('x2', [i**2 for i in range(4)])
    >>> for kwarg, log in iterator:
    ...     print(kwarg, log)
    {'x2': 0} {'x2': 0}
    {'x2': 1} {'x2': 1}
    {'x2': 4} {'x2': 4}
    {'x2': 9} {'x2': 9}

    You can do the same thing by iterating over integers and apply a square method.

    >>> iterator = Iterator('x2', range(4), 'x', lambda x: x**2)
    >>> for kwarg, log in iterator:
    ...     print(kwarg, log)
    {'x2': 0} {'x': 0}
    {'x2': 1} {'x': 1}
    {'x2': 4} {'x': 2}
    {'x2': 9} {'x': 3}
    """
    def __init__(self, parameter, values, name=None, process=None):
        self.parameter = parameter
        self.values = values
        if name is None:
            name = parameter
        self.name = name
        if process is None:
            process = do_nothing
        self.process = process

    def __iter__(self):
        for v in self.values:
            yield {self.parameter: self.process(v)}, {self.name: v}

    def __len__(self):
        return len(self.values)


class SingleXP:
    def __init__(self, name, **params):
        self.name = name
        self.iterator = params.pop('iterator', None)
        self.params = params

    def __len__(self):
        if self.iterator is None:
            return 1
        else:
            return len(self.iterator)

    def __iter__(self):
        if self.iterator is None:
            yield self.name, None, self.params
        else:
            for kwarg, log, in self.iterator:
                yield self.name, log, {**self.params, **kwarg}


class XP:
    def __init__(self, name, **params):
        self.xp_list = [SingleXP(name, **params)]

    def __iter__(self):
        for xp in self.xp_list:
            for x in xp:
                yield x

    def __add__(self, other):
        res = XP(name=None)
        res.xp_list = self.xp_list + other.xp_list
        return res

    def __radd__(self, other):
        if other == 0:
            return self
        return self+other

    def __len__(self):
        return sum(len(xp) for xp in self.xp_list)


def regret_delay(model):
    simu = model.simulator
    regret = simu.compute_regret()
    delay = sum(simu.compute_average_queues())
    return {'regret': regret, 'delay': delay}


class Runner:
    def __init__(self, extractor=None):
        if extractor is None:
            self.extractor = regret_delay
        else:
            self.extractor = extractor

    def __call__(self, tup):
        name, kv, params = tup
        params = {**params}
        model = params.pop('model')
        model.run(**params)
        return name, kv, self.extractor(model)


def aggregate(results):
    """
    Parameters
    ----------
    results: :class:`list`
        Computed results. Each entry is a dictionary associated to a given run.
    Returns
    -------
    :class:`dict`
        All results are gathered by experiment name, then by varying input / metric if applicable.
    """
    res = dict()
    for name, kv, r in results:
        if kv is None:
            res[name] = r
            continue
        rr = {**kv, **r}
        if name not in res:
            res[name] = {k: [] for k in rr}
        for k, v in rr.items():
            res[name][k].append(v)
    return res


def cached(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, cache_name=None, cache_path='.', cache_overwrite=False, **kwargs):
        if cache_name is not None:
            cache = Path(cache_path) / Path(f"{cache_name}.pkl.gz")
            if cache.exists() and not cache_overwrite:
                with gzip.open(cache, 'rb') as f:
                    return pickle.load(f)
        else:
            cache = None
        res = func(*args, **kwargs)
        if cache is not None:
            with gzip.open(cache, 'wb') as f:
                pickle.dump(res, f)
        return res
    return wrapper_decorator


@cached
def evaluate(xps, extractor=None, pool=None):
    """
    Examples
    --------

    >>> import stochastic_matching as sm
    >>> import numpy as np
    >>> diamond = sm.CycleChain()
    >>> base = {'model': diamond, 'n_steps': 1000, 'seed': 42, 'rewards': [1, 2.9, 1, -1, 1]}
    >>> xp = XP('Diamond', simulator='longest', **base)
    >>> evaluate(xp)
    {'Diamond': {'regret': 0.08799999999999886, 'delay': 1.634}}
    >>> xp1 = XP('e-filtering', simulator='e_filtering', **base,
    ...          iterator=Iterator('epsilon', [.01, .1, 1], name='e'))
    >>> xp2 = XP(name='k-filtering', simulator='longest', forbidden_edges=True,
    ...          iterator=Iterator('k', [0, 10, 100]), **base)
    >>> xp3 = XP(name='egpd', simulator='virtual_queue',
    ...          iterator=Iterator('beta', [.01, .1, 1]), **base)
    >>> xp = sum([xp1, xp2, xp3])
    >>> len(xp)
    9
    >>> import multiprocess as mp
    >>> with mp.Pool(processes=2) as p:
    ...     res = evaluate(xp, pool=p)
    >>> for name, r in res.items():
    ...     print(name)
    ...     for k, v in r.items():
    ...         print(f"{k}: {np.array(v)}")
    e-filtering
    e: [0.01 0.1  1.  ]
    regret: [0.002 0.017 0.103]
    delay: [10.538  6.95   1.952]
    k-filtering
    k: [  0  10 100]
    regret: [ 8.8000000e-02  2.0000000e-03 -8.8817842e-16]
    delay: [ 1.634  7.342 13.542]
    egpd
    beta: [0.01 0.1  1.  ]
    regret: [0.003 0.043 0.076]
    delay: [92.024 10.13   1.888]
    """
    if extractor is None:
        extractor = regret_delay
    runner = Runner(extractor)
    if pool is None:
        res = [runner(p) for p in tqdm(xps)]
    else:
        res = tqdm(pool.imap(runner, xps), total=len(xps))
    return aggregate(res)


# from tqdm import tqdm
# from copy import deepcopy
#
#
# class VariableParameter:
#     """
#     Allows to vary one single parameter of simulation for experiments.
#
#     Parameters
#     ----------
#     model: :class:`~stochastic_matching.model.Model`
#         Model to simulate.
#     name: :class:`str`
#         Name of the experiment.
#     key: :class:`str`
#         Variable parameter.
#     values: iterable
#         Values of the parameter.
#     kwargs: :class:`dict`
#         All other keyword parameters will be passed to simulation (including the choice of `simulator`).
#     """
#     def __init__(self, model, name, key, values, **kwargs):
#         self.name = name
#         self.key = key
#         self.values = values
#         self.model = deepcopy(model)
#         self.model.simulator = None
#         self.kwargs = kwargs
#
#     def __iter__(self):
#         for v in self.values:
#             params = {**self.kwargs, self.key: v}
#             yield self.name, self.key, params, self.model
#
#
# def build_metric_computer(metric_extractor=None):
#     """
#     Parameters
#     ----------
#     metric_extractor: callable, optional
#         The metric extractor must have a (params, model) signature and return a dictionary of metrics.
#
#     Returns
#     -------
#     callable
#         A function with signature (name, key, params, model).
#     """
#     if metric_extractor is None:
#         metric_extractor = regret_delay
#
#     def compute(name, key, params, model):
#         model.run(**params)
#         return {'name': name, key: params[key], **metric_extractor(model=model, params=params)}
#
#     return compute
#
#
# def regret_delay(model, params):
#     """
#     Parameters
#     ----------
#     model: :class:`~stochastic_matching.model.Model`
#         Model simulated.
#     params: :class:`dict`
#         Parameters of the simulation
#
#     Returns
#     -------
#     :class:`dict`
#         Regret and delay.
#     """
#     simu = model.simulator
#     regret = simu.compute_regret()
#     delay = sum(simu.compute_average_queues())
#     return {'regret': regret, 'delay': delay}
#
#
# def aggregate(results):
#     """
#     Parameters
#     ----------
#     results: :class:`list`
#         Computed results. Each entry is a dictionary associated to a given run.
#     Returns
#     -------
#     :class:`dict`
#         All results are gathered by experiment name, then by varying input / metric.
#     """
#     res = dict()
#     for r in results:
#         name = r.pop('name')
#         if name not in res:
#             res[name] = {k: [] for k in r}
#         for k, v in r.items():
#             res[name][k].append(v)
#     return res
#
#
# def evaluate(xps, metric_extractor=None, pool=None):
#     """
#     Parameters
#     ----------
#     xps: :class:`~stochastic_matching.simulator.parallel.VariableParameter` or :class:`list`
#         Experiment(s) to run.
#     metric_extractor: callable, optional.
#         The metric extractor must have a (params, model) signature and return a dictionary of metrics.
#         Default to computing regret and delay.
#     pool: :class:`~multiprocess.pool.Pool`, optional.
#         Existing pool of workers.
#
#     Returns
#     -------
#     :class:`dict`
#         Result of the experiment(s).
#
#     Examples
#     --------
#
#     >>> import stochastic_matching as sm
#     >>> import numpy as np
#     >>> diamond = sm.CycleChain()
#     >>> base = {'model': diamond, 'n_steps': 1000, 'seed': 42, 'rewards': [1, 2.9, 1, -1, 1]}
#     >>> xp1 = VariableParameter(name='e-filtering', simulator='e_filtering',
#     ...                         key='epsilon', values=[.01, .1, 1], **base)
#     >>> xp2 = VariableParameter(name='k-filtering', simulator='longest', forbidden_edges=True,
#     ...                         key='k', values=[0, 10, 100], **base)
#     >>> xp3 = VariableParameter(name='egpd', simulator='virtual_queue',
#     ...                         key='beta', values=[.01, .1, 1], **base)
#     >>> res = evaluate(xp1)
#     >>> for k, v in res['e-filtering'].items():
#     ...     print(f"{k}: {np.array(v)}")
#     epsilon: [0.01 0.1  1.  ]
#     regret: [0.002 0.017 0.103]
#     delay: [10.538  6.95   1.952]
#     >>> import multiprocess as mp
#     >>> with mp.Pool(processes=2) as p:
#     ...     res = evaluate([xp2, xp3], pool=p)
#     >>> for name, r in res.items():
#     ...     print(name)
#     ...     for k, v in r.items():
#     ...         print(f"{k}: {np.array(v)}")
#     k-filtering
#     k: [  0  10 100]
#     regret: [ 8.8000000e-02  2.0000000e-03 -8.8817842e-16]
#     delay: [ 1.634  7.342 13.542]
#     egpd
#     beta: [0.01 0.1  1.  ]
#     regret: [0.003 0.043 0.076]
#     delay: [92.024 10.13   1.888]
#     """
#     if isinstance(xps, list):
#         jobs = [x for xp in xps for x in xp]
#     else:
#         jobs = [x for x in xps]
#     compute = build_metric_computer(metric_extractor)
#     if pool is None:
#         res = [compute(*args) for args in tqdm(jobs)]
#     else:
#         res = tqdm(pool.imap(lambda args: compute(*args), jobs), total=len(jobs))
#     return aggregate(res)
