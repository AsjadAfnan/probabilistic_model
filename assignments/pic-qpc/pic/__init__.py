"""Tree-PIC to QPC PyTorch library with analytic Gaussian baseline.

This library implements Probabilistic Inference Circuits (PICs) with static quadrature
for continuous latent variables, providing both analytic and neural conditional families.
"""

from .structures import LatentTree, TreeSpec
from .quadrature import Quadrature
from .compile import TreePIC, QPC
from .leaves import GaussianLeaf, BernoulliLeaf
from .conditionals import LinearGaussian, NeuralEnergyConditional
from .evaluate import log_prob, marginal_log_prob, most_probable_explanation

__version__ = "0.1.0"
__all__ = [
    "LatentTree",
    "TreeSpec", 
    "Quadrature",
    "TreePIC",
    "QPC",
    "GaussianLeaf",
    "BernoulliLeaf",
    "LinearGaussian",
    "NeuralEnergyConditional",
    "log_prob",
    "marginal_log_prob",
    "most_probable_explanation",
]
