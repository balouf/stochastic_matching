"""Top-level package for Stochastic Matching."""

__author__ = """Fabien Mathieu"""
__email__ = 'fabien.mathieu@normalesup.org'
__version__ = '0.2.2'

from stochastic_matching.model import Model
from stochastic_matching.graphs import Path, Star, Cycle, Codomino, CycleChain, Complete, Pyramid, HyperPaddle, \
    KayakPaddle, Lollipop, Tadpole, Barbell, Fan, concatenate
from stochastic_matching.old_simulator.age_based import FCFM
from stochastic_matching.old_simulator.virtual_queue import VQSimulator
from stochastic_matching.old_simulator.size_based import LongestSimulator, PrioritySimulator, QueueSizeSimulator, \
    RandomItemSimulator, \
    RandomNodeSimulator, FilteringGreedy
