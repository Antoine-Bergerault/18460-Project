from dataclasses import dataclass
from enum import Enum
import numpy as np
import numpy.typing as npt
import server
from typing import Any, Callable

Computation = Enum('Computation', ['HIGH', 'LOW'])

@dataclass
class ClientParameters:
    # TODO: complete and add doc
    
    server: server.Server
    id: int

    loss: Callable[[npt.ArrayLike, Any], Any]
    loss_grad: Callable[[npt.ArrayLike, Any], Any]
    loss_hessian: Callable[[npt.ArrayLike, Any], Any]
    
    initial_guess: npt.ArrayLike

    penalty: float

class Client():
    def __init__(self, dataset : Any, params: ClientParameters, computation: Computation = Computation.HIGH):
        self.dataset = dataset

        self.params = params

        self.loss = params.loss
        self.loss_grad = params.loss_grad
        self.loss_hessian = params.loss_hessian
        
        self.computation = computation

        self.primals = params.initial_guess
        self.duals = np.zeros_like(params.initial_guess) # might be changed to a better initial guess for duals

        self.penalty = params.penalty

    def update(self, consensus):
        if self.computation == Computation.HIGH:
            self.second_order_update(consensus)
        else:
            self.first_order_update(consensus)

        # TODO: send values to server
        # Implement it concurrently

        # Non-concurrent version:
        self.params.server.update_client(self.params.id, self.primals, self.duals)

    # TODO: implement second order update
    # Note: We should avoid taking the inverse of a matrix
    # (solve linear system, can be done by the means of a decomposition)
    #
    # Can be decoupled from the class in the future to test different
    # algorithms
    def second_order_update(self, consensus):
        pass

    # TODO: implement first order update
    #
    # Can be decoupled from the class in the future to test different
    # algorithms
    #
    # Implementation comes directly from the paper
    def first_order_update(self, consensus):
        a = 0.02 # TODO: change this to a non-arbitrary value
        b = 0.02 # TODO: change this to a non-arbitrary value

        old_primals = self.primals

        self.primals = self.primals - a*(self.loss_grad(self.primals, self.dataset) - self.duals + self.penalty * (self.primals - consensus))
        self.duals = self.duals + b*(consensus - old_primals)