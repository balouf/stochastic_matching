import numpy as np
from stochastic_matching.common import class_converter


class Metric:
    """
    Blueprint for extracting a metric from the simulator.
    """
    name = None

    @staticmethod
    def get(simu):
        """
        Parameters
        ----------
        simu: :class:`~stochastic_matching.simulator.simulator.Simulator`
            Simulator to use.

        Returns
        -------
        Object
            A metric.
        """
        raise NotImplementedError


class Traffic(Metric):
    """
    Matched edges.
    """
    name = 'traffic'

    @staticmethod
    def get(simu):
        return simu.logs.traffic


class StepsDone(Metric):
    """
    Simulation duration.
    """
    name = 'steps_done'

    @staticmethod
    def get(simu):
        return simu.logs.steps_done


class Income(Metric):
    """
    Number of arrived items per class.
    """
    name = 'income'

    @staticmethod
    def get(simu):
        return simu.logs.income


class Regret(Metric):
    """
    Given the matched items observed, difference between the best possible match
    between these items and the actual match.

    Notes
    -----

    * Regret depends on the assigned rewards.
    * Unmatched items are ignored to avoid side effects (e.g. if the optimal allocation is
      unstable, for example because of negative weights).
    """
    name = 'regret'

    @staticmethod
    def get(simu):
        rewards = getattr(simu, 'rewards', np.ones(simu.model.m, dtype=int))
        original_rates = simu.model.rates
        flow = Flow.get(simu)
        simu.model.rates = simu.model.incidence @ flow
        best_flow = simu.model.optimize_rates(rewards)
        simu.model.rates = original_rates
        return rewards @ (best_flow - flow)


class AvgQueues(Metric):
    """
    Average queue size per item class.
    """
    name = 'avg_queues'

    @staticmethod
    def get(simu):
        return (simu.logs.queue_log @ np.arange(simu.max_queue)) / simu.logs.steps_done


class Delay(Metric):
    """Average waiting time (i.e. average total queue size divided by global arrival rate)."""
    name = 'delay'

    @staticmethod
    def get(simu):
        return np.sum(AvgQueues.get(simu)) / np.sum(simu.model.rates)


class CCDF(Metric):
    """
    Complementary Cumulative Distribution Function of queue size for each item class.
    """
    name = 'ccdf'

    @staticmethod
    def get(simu):
        steps = simu.logs.steps_done
        n = simu.model.n
        return (steps - np.cumsum(np.hstack([np.zeros((n, 1)), simu.logs.queue_log]), axis=1)) / steps


class Flow(Metric):
    """
    Edge matching rate.
    """
    name = 'flow'

    @staticmethod
    def get(simu):
        tot_mu = np.sum(simu.model.rates)
        steps = simu.logs.steps_done
        return simu.logs.traffic * tot_mu / steps


def get_metrics(simu, metrics):
    """
    Extraction of metrics from a simulator. Metrics can be:

    * Name of a :class:`~stochastic_matching.simulator.metrics.Metric` subclass
    * A :class:`~stochastic_matching.simulator.metrics.Metric` subclass
    * A callable that takes a :class:`~stochastic_matching.simulator.simulator.Simulator` as input
      (the name will be the name of the callable)
    * A list of metrics defined as above

    Parameters
    ----------
    simu: :class:`~stochastic_matching.simulator.simulator.Simulator`
        Simulator to use.
    metrics: :class:`str` or :class:`~stochastic_matching.simulator.metrics.Metric` or :class:`list` or callable.
        Metric(s) to extract.
    Returns
    -------
    :class:`dict`
        Metric(s).

    """
    if isinstance(metrics, list):
        return {k: v for metric in metrics
                for k, v in get_metrics(simu, metric).items()}
    try:
        getter = class_converter(metrics, Metric)
        return {getter.name: getter.get(simu)}
    except TypeError:
        return {metrics.__name__: metrics(simu)}
