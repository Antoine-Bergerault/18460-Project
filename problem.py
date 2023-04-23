from dataclasses import dataclass, field
import numpy.typing as npt
from typing import Any, Callable

@dataclass
class OptimizationProblem:
    # TODO: complete and add doc
    
    tol: float
    ctol: float
    max_iter: int

    # The objective function is the sum of individual losses

    loss: Callable[[npt.ArrayLike, Any, dict], Any]
    loss_grad: Callable[[npt.ArrayLike, Any, dict], Any]
    loss_hessian: Callable[[npt.ArrayLike, Any, dict], Any]

    hyper_parameters: dict = field(default_factory={
        "penalty": 2, # penalty for augmented Lagrangian
        "x0": None # value of the initial guess
    })