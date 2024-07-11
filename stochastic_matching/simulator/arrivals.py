import numpy as np
from numba import njit, int64, float64
from numba.experimental import jitclass


@njit
def create_prob_alias(mu):
    """
    Prepare vector to draw a distribution with the alias method.

    Based on https://www.keithschwarz.com/darts-dice-coins/.

    Parameters
    ----------
    mu: :class:`list` or :class:`~numpy.ndarray`
        Arrival intensities.

    Returns
    -------
    prob: :class:`~numpy.ndarray`
        Probabilities to stay in the drawn bucket
    alias: :class:`~numpy.ndarray`
        Redirection array

    Examples
    --------

    >>> probas, aliases = create_prob_alias([2 ,2, 3, 1])
    >>> probas
    array([1. , 1. , 1. , 0.5])
    >>> aliases.astype(int)
    array([0, 0, 0, 2])
    """
    if isinstance(mu, list):
        cmu = np.array(mu)
    else:
        cmu = mu
    n = len(cmu)
    alias = np.zeros(n, dtype=np.int64)
    prob = np.zeros(n)
    # noinspection PyUnresolvedReferences
    normalized_intensities = cmu * n / np.sum(cmu)
    small = [i for i in range(n) if normalized_intensities[i] < 1]
    large = [i for i in range(n) if normalized_intensities[i] >= 1]
    while small and large:
        l, g = small.pop(), large.pop()
        prob[l], alias[l] = normalized_intensities[l], g
        normalized_intensities[g] += normalized_intensities[l] - 1
        if normalized_intensities[g] < 1:
            small.append(g)
        else:
            large.append(g)
    for i in large + small:
        prob[i] = 1
    return prob, alias


@jitclass
class Arrivals:
    """
    Parameters
    ----------
    mu: :class:`list` or :class:`~numpy.ndarray`
        Arrival intensities.
    seed: :class:`int`
        Seed for random generator.

    Examples
    --------

    >>> arrivals = Arrivals([2 ,2, 3, 1], seed=42)
    >>> arrivals.prob
    array([1. , 1. , 1. , 0.5])
    >>> arrivals.alias.astype(int)
    array([0, 0, 0, 2])
    >>> from collections import Counter
    >>> Counter([arrivals.draw() for _ in range(800)])
    Counter({2: 291, 1: 210, 0: 208, 3: 91})
    """
    prob: float64[:]
    alias: int64[:]
    n: int

    def __init__(self, mu, seed=None):
        self.prob, self.alias = create_prob_alias(mu)
        self.n = len(mu)
        if seed is not None:
            self.set_seed(seed)

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)

    def draw(self):
        node = np.random.randint(self.n)
        if np.random.rand() > self.prob[node]:
            node = self.alias[node]
        return node
