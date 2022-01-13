"""Top-level package for Stochastic Matching."""

__author__ = """Fabien Mathieu"""
__email__ = 'fabien.mathieu@normalesup.org'
__version__ = '0.1.0'

from stochastic_matching.stochastic_matching import MQ
from stochastic_matching.graphs.classes import SimpleGraph, HyperGraph
from stochastic_matching.graphs.generators import tadpole_graph, bicycle_graph, kayak_paddle_graph, triangle_chain, fan, hyper_paddle
