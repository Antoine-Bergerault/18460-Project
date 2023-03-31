from enum import Enum

Computation = Enum('Computation', ['HIGH', 'LOW'])

class Client():
    def __init__(self, dataset, computation: Computation = Computation.HIGH):
        self.dataset = dataset
        self.computation = computation
        pass

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