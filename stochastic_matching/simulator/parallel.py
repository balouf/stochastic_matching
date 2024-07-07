from joblib import Parallel, delayed
from tqdm import tqdm


class VariableParameter:
    def __init__(self, model, name, key, values, **kwargs):
        self.name = name
        self.key = key
        self.values = values
        self.model = model
        self.kwargs = kwargs

    def __iter__(self):
        for v in self.values:
            params = {**self.kwargs, self.key: v}
            yield self.name, self.key, params, self.model


def build_metric_computer(metric_extractor=None):
    if metric_extractor is None:
        metric_extractor = regret_delay

    def compute(name, key, params, model):
        model.run(**params)
        return {'name': name, key: params[key], **metric_extractor(params, model)}

    return compute


def regret_delay(params, model):
    rewards = params['rewards']
    best_flow = model.optimize_rates(rewards)
    flow = model.simulator.compute_flow()
    regret = rewards @ (best_flow - flow)
    delay = sum(model.simulator.compute_average_queues())
    return {'regret': regret, 'delay': delay}


def aggregate(results):
    res = dict()
    for r in results:
        name = r.pop('name')
        if name not in res:
            res[name] = {k: [] for k in r}
        for k, v in r.items():
            res[name][k].append(v)
    return res


def evaluate(xps, n_jobs=-1, metric_extractor=None):
    if isinstance(xps, list):
        jobs = [x for xp in xps for x in xp]
    else:
        jobs = [x for x in xps]
    compute = build_metric_computer(metric_extractor)
    res = Parallel(n_jobs=n_jobs)(delayed(compute)(*args) for args in tqdm(jobs))
    return aggregate(res)
