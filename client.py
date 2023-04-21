from dataclasses import dataclass
from enum import Enum
import numpy as np
import numpy.typing as npt
from typing import Any, Callable

Computation = Enum('Computation', ['HIGH', 'LOW'])

@dataclass
class ClientParameters:
    # TODO: complete and add doc
    
    server: 'server.Server' # forward reference to avoid circular dependency
    id: int

    loss: Callable[[npt.ArrayLike, Any], Any]
    loss_grad: Callable[[npt.ArrayLike, Any], Any]
    loss_hessian: Callable[[npt.ArrayLike, Any], Any]
    
    initial_guess: npt.ArrayLike

    penalty: float

    lr: Callable[[int], float]

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

    def update(self, consensus, k):
        if self.computation == Computation.HIGH:
            self.second_order_update(consensus, k)
        else:
            self.first_order_update(consensus, k)

        # Non-concurrent version:
        #self.params.server.update_client(self.params.id, self.primals, self.duals)

    # Can be decoupled from the class in the future to test different
    # algorithms
    #
    # Implementation comes directly from the paper with some computational tricks added
    def second_order_update(self, consensus, k):
        # we use same learning rate for both primals and duals
        a = self.params.lr(k)
        b = self.params.lr(k)

        old_primals = self.primals

        lagrangian_hessian = self.loss_hessian(self.primals, self.dataset) + self.params.penalty * np.eye(self.primals.shape[0])
        
        # We check the positive definiteness of the hessian
        # If not PD, we regularize it and perform a damped Newton update
        #
        # To check if a matrix is PD, we try to compute its cholesky factorization
        # It is faster than the naive check on eigenvalues

        def regularize(H, beta):
            try:
                np.linalg.cholesky(H)
                return H
            except np.linalg.linalg.LinAlgError:
                return regularize(H + beta*np.eye(H.shape[0]), beta)

        lagrangian_hessian = regularize(lagrangian_hessian, 0.01)
        
        first_order_step = self.loss_grad(self.primals, self.dataset) - self.duals + self.params.penalty * (self.primals - consensus)

        delta_x = np.linalg.solve(lagrangian_hessian, -a*first_order_step) # x_k+1 - x_k = -a * H^-1 @ first_order_step
        self.primals = delta_x + self.primals

        self.duals = self.duals + b*lagrangian_hessian @ (consensus - old_primals)

    # Can be decoupled from the class in the future to test different
    # algorithms
    #
    # Implementation comes directly from the paper
    def first_order_update(self, consensus, k):
        # we use same learning rate for both primals and duals
        a = self.params.lr(k)
        b = self.params.lr(k)

        old_primals = self.primals

        self.primals = self.primals - a*(self.loss_grad(self.primals, self.dataset) - self.duals + self.params.penalty * (self.primals - consensus))
        self.duals = self.duals + b*(consensus - old_primals)