from tqdm import tqdm
from pathlib import Path
import gzip
import dill as pickle
import functools

from stochastic_matching.simulator.metrics import get_metrics


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
    log: :class:`dict`
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
    """
    Produce a single experiment, with possibly one variable parameter.

    Parameters
    ----------
    name: :class:`str`
        Name of the experiment.
    iterator: :class:`~stochastic_matching.simulator.parallel.Iterator`, optional
        Argument that varies.
    params: :class:`dict`, optional
        Various (fixed) arguments.
    """
    def __init__(self, name, iterator=None, **params):
        self.name = name
        self.iterator = iterator
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
    """
    Produce an experiment, with possibly one variable parameter.
    Addition can be used to concatenate experiments.

    Parameters
    ----------
    name: :class:`str`
        Name of the experiment.
    iterator: :class:`~stochastic_matching.simulator.parallel.Iterator`, optional
        Argument that varies.
    params: :class:`dict`, optional
        Various (fixed) arguments.
    """

    def __init__(self, name, iterator=None, **params):
        self.xp_list = [SingleXP(name, iterator, **params)]

    def __iter__(self):
        for xp in self.xp_list:
            for x in xp:
                yield x

    def __add__(self, other):
        if other == 0:
            return self
        res = XP(name=None)
        res.xp_list = self.xp_list + other.xp_list
        return res

    def __radd__(self, other):
        return self+other

    def __len__(self):
        return sum(len(xp) for xp in self.xp_list)


def regret_delay(model):
    """
    Default metric extractor.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Simulated model.


    Returns
    -------
    :class:`dict`
        Regret and delay.
    """
    simu = model.simulator
    regret = simu.compute_regret()
    delay = sum(simu.compute_average_queues())
    return {'regret': regret, 'delay': delay}


class Runner:
    """
    Parameters
    ----------
    metrics: :class:`list`
        Metrics to extract.
    """

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, tup):
        name, kv, params = tup
        params = {**params}
        model = params.pop('model')
        model.run(**params)
        out = get_metrics(model.simulator, self.metrics)
        # logs = model.simulator.logs
        # out = {k: getattr(logs, k) for k in XP_LOGS}
        # if self.callback:
        #     out.update(self.callback(model))
        return name, kv, out


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
    """
    Enables functions that take a lot of time to compute to store their output on a file.

    If the original function is called with an additional parameter `cache_name`, it will look for a file named
    `cache_name.pkl.gz`. If it exists, it returns its content, otherwise it computes the results and
    stores it is returned.

    Additional parameter `cache_path` indicates where to look the file (default to current directory).

    Additional parameter `cache_overwrite` forces to always compute the function even if the file exists.
    Default to False.

    Parameters
    ----------
    func: callable
        Function to make cachable.

    Returns
    -------
    callable
        The function with three additional arguments: `cache_name`, `cache_path`, `cache_overwrite`.

    Examples
    --------

    >>> from tempfile import TemporaryDirectory as TmpDir
    >>> @cached
    ... def f(x):
    ...     return x**3
    >>> with TmpDir() as tmp:
    ...     content_before = [f.name for f in Path(tmp).glob('*')]
    ...     res = f(3, cache_name='cube', cache_path=Path(tmp))
    ...     content_after = [f.name for f in Path(tmp).glob('*')]
    ...     res2 = f(3, cache_name='cube', cache_path=Path(tmp)) # This does not run f!
    >>> res
    27
    >>> res2
    27
    >>> content_before
    []
    >>> content_after
    ['cube.pkl.gz']
    """
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
def evaluate(xps, metrics=None, pool=None):
    """
    Evaluate some experiments. Results can be cached.

    Parameters
    ----------
    xps: :class:`~stochastic_matching.simulator.parallel.XP`
        Experiment(s) to run.
    metrics: :class:`list`, optional.
        Metrics to get.
    pool: :class:`~multiprocess.pool.Pool`, optional.
        Existing pool of workers.

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
    >>> xp = XP('Diamond', simulator='longest', **base)
    >>> res = evaluate(xp, ['income', 'steps_done', 'regret', 'delay'])
    >>> for name, r in res.items():
    ...     print(name)
    ...     for k, v in r.items():
    ...         print(f"{k}: {np.round(v, 4)}")
    Diamond
    income: [216 304 293 187]
    steps_done: 1000
    regret: 0.088
    delay: 0.1634
    >>> def q0(simu):
    ...     return simu.avg_queues[0]
    >>> res = evaluate(xp, ['flow', 'avg_queues', q0])
    >>> res['Diamond']['flow']
    array([1.19, 0.97, 0.97, 0.88, 0.99])
    >>> res['Diamond']['avg_queues']
    array([0.531, 0.214, 0.273, 0.616])
    >>> res['Diamond']['q0']
    0.531
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
    ...     res = evaluate(xp, ['regret', 'delay'], pool=p)
    >>> for name, r in res.items():
    ...     print(name)
    ...     for k, v in r.items():
    ...         print(f"{k}: {np.array(v)}")
    e-filtering
    e: [0.01 0.1  1.  ]
    regret: [0.002 0.017 0.103]
    delay: [1.0538 0.695  0.1952]
    k-filtering
    k: [  0  10 100]
    regret: [ 8.8000000e-02  2.0000000e-03 -8.8817842e-16]
    delay: [0.1634 0.7342 1.3542]
    egpd
    beta: [0.01 0.1  1.  ]
    regret: [0.003 0.043 0.076]
    delay: [9.2024 1.013  0.1888]
    """
    if metrics is None:
        metrics = ['regret', 'delay']
    runner = Runner(metrics=metrics)
    if pool is None:
        res = [runner(p) for p in tqdm(xps)]
    else:
        res = tqdm(pool.imap(runner, xps), total=len(xps))
    return aggregate(res)
