"""Top-level package for Stochastic Matching."""

__author__ = """Fabien Mathieu"""
__email__ = 'fabien.mathieu@normalesup.org'
__version__ = '0.3.3'

from stochastic_matching.model import Model
from stochastic_matching.graphs import Path, Star, Cycle, Codomino, CycleChain, Complete, Pyramid, HyperPaddle, \
    KayakPaddle, Lollipop, Tadpole, Barbell, Fan, concatenate
from stochastic_matching.simulator.simulator import Simulator
from stochastic_matching.simulator.fcfm import FCFM
from stochastic_matching.simulator.longest import Longest
from stochastic_matching.simulator.priority import Priority
from stochastic_matching.simulator.random_edge import RandomEdge
from stochastic_matching.simulator.random_item import RandomItem
from stochastic_matching.simulator.virtual_queue import VirtualQueue
from stochastic_matching.simulator.e_filtering import EFiltering
from stochastic_matching.simulator.parallel import XP, Runner, Iterator, evaluate
