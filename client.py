from enum import Enum
import numpy.typing as npt
from typing import Any, Callable

Computation = Enum('Computation', ['HIGH', 'LOW'])

class Client():
    def __init__(self, dataset : npt.ArrayLike, loss: Callable[[npt.ArrayLike, dict], Any], loss_gd: Callable[[npt.ArrayLike, dict], Any], 
                 loss_hs: Callable[[npt.ArrayLike, dict], Any], computation: Computation = Computation.HIGH):
        self.dataset = dataset
        
        self.loss = loss
        self.loss_grad = loss_gd
        self.loss_hessian = loss_hs
        
        self.computation = computation

    def update(self):
        if self.computation == Computation.HIGH:
            self.second_order_update()
        else:
            self.first_order_update()

    # TODO: implement second order update
    # Note: We should avoid taking the inverse of a matrix
    # (solve linear system, can be done by the means of a decomposition)
    #
    # Can be decoupled from the class in the future to test different
    # algorithms
    def second_order_update(self):
        pass

    # TODO: implement first order update
    #
    # Can be decoupled from the class in the future to test different
    # algorithms
    def first_order_update(self):
        pass