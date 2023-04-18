from collections import namedtuple
from functools import cached_property
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from problem import OptimizationProblem
from tasks.task import Task

class LogisticRegressionTask(Task):
    def __init__(self, config: Config = default_config) -> None:
        super().__init__(config)

        self.lr = config.lr
        self.clients = config.clients
        self.number = config.number
        self.lb = config.lb
        self.ub = config.ub

        self.optimizer = config.optimizer

    @cached_property
    def dataset(self):
        if self.optimizer is None or not isinstance(self.optimizer, np.ndarray) or not (self.optimizer.shape == (2,) or self.optimizer.shape == (2,1)):
            raise ValueError("Cannot generate dataset without setting a correct optimizer. It should be a (2,) or (2,1) numpy array")

        # TODO: Determine that

        pass
        
    def get_partitions(self):
        # TODO: determine that
        pass

    def get_problem(self):
        hyper_parameters = {
            "penalty": 10,
            "x0": np.random.standard_normal((2, 1)), # one multiplicative term and one bias term
            "regularization_factor": 2
        }

        # simple linear regression cost (mean squared error loss)
        def cost(x, dataset, params):
            categories = dataset[:, 0]
            
            features = dataset[:, 1:-1]

            # sigmoid of -Aw
            prediction = 1/(1 + np.exp(-features @ x))
            
            return -categories.T @ np.log(prediction) - (1 - categories).T @ np.log(1 - prediction) + (params["regularization_factor"] / 2) * x.T @ x

        def cost_grad(x, dataset, params):
            categories = dataset[:, 0]
            
            features = dataset[:, 1:-1]

            # sigmoid of -Aw
            prediction = 1/(1 + np.exp(-features @ x))
            grad = np.sum(features * (prediction - categories), axis=0) + params["regularization_factor"] * x

            return grad[:, None] # make sure to return something with shape (2, 1)

        def cost_hessian(x, dataset, params):
            categories = dataset[:, 0]
            
            features = dataset[:, 1:-1]

            # sigmoid of -Aw
            prediction = 1/(1 + np.exp(-features @ x))
            hessian = features * np.diag(prediction * (1 - prediction)) @ features.T

            return hessian

        problem = OptimizationProblem(tol=1e-6, ctol=1e-6, max_iter=20000, lr=self.lr, loss=cost, loss_grad=cost_grad, 
                                      loss_hessian=cost_hessian, hyper_parameters=hyper_parameters)
        
        return problem

    def visualize(self):
        pass

    def visualize_solution(self, solution):
        pass