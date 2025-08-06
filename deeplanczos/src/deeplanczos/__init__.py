"""
Lanczos diagonalization for Neural networks
"""

__version__ = "0.1.0"
__author__ = "Luca Di Carlo"
__email__ = "lucadc@princeton.edu"

from .utils import *
from .power_method import dot_product, InitPowerMethodVector, orthogonlize
from .power_method import ComputeHVP, initializeHVP
from .power_method import * 

def hello(): 
    print("this is a library for Lanczos diagonalization for Neural Networks in jax")
    return 0 


__all__ = [
    "GenerateRandomVectors",
    "dot_product",
    "InitPowerMethodVector",
    "orthogonlize",
    "initializeHVP",
    "ComputeHVP",
    "accumulateHVP",
    "PowerMethodIterate"

]
