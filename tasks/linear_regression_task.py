'''
Linear Regression Task with Mean Squared Error loss
'''

from client import Computation
from dataclasses import dataclass
from functools import cached_property
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from problem import OptimizationProblem
from tasks.task import Task, Config

@dataclass
class LGRTConfig(Config):
    number: int
    lb: float
    ub: float
    optimizer: npt.ArrayLike

default_config = LGRTConfig(clients=[
    *(3*(Computation.HIGH,)),
    *(2*(Computation.LOW,))
], number=200, lb=0, ub=100, optimizer=np.array([2, 30]), lr=lambda k:0.1/sqrt(k), nlr=0.9)

solo_config = LGRTConfig(clients=[
    Computation.HIGH
], number=200, lb=0, ub=100, optimizer=np.array([2, 30]), lr=0.01, nlr=0.9)

class LinearRegressionTask(Task):
    def __init__(self, config: LGRTConfig = default_config) -> None:
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

        x = np.linspace(self.lb, self.ub, num=self.number)
        y = self.optimizer[0]*x + self.optimizer[1]

        points = np.stack((x, y), axis=-1)

        return points + 4*np.random.standard_normal(points.shape) # add noise to points
        
    def get_subsets(self):
        high = np.max(self.dataset)
        normalized = (self.dataset + (high - 1) * np.array([0, self.optimizer[1]])) / high
        # strange normalization that preserves minimizer
        return normalized.reshape((len(self.clients), -1, 2))

    def get_problem(self):
        hyper_parameters = {
            "penalty": 10,
            "x0": np.random.standard_normal((2, 1)), # one multiplicative term and one bias term
            "normalization_factor": len(self.dataset)
        }

        # simple linear regression cost (mean squared error loss)
        def cost(x, dataset, params):
            m = x[0]
            b = x[1]
            
            preimages = dataset[:, 0] # x-coordinate of points
            images = dataset[:, 1] # y-coordinate of points
            
            prediction = m*preimages + b
            
            return np.sum((prediction - images)**2) / params["normalization_factor"]

        def cost_grad(x, dataset, params):
            m = x[0]
            b = x[1]
            
            preimages = dataset[:, 0] # x-coordinate of points
            images = dataset[:, 1] # y-coordinate of points
            
            prediction = m*preimages + b
            
            grad = np.array([
                np.sum(2*preimages*(prediction - images)),
                np.sum(2*(prediction - images))
            ]) / params["normalization_factor"]
            
            return grad[:, None] # make sure to return something with shape (2, 1)

        def cost_hessian(x, dataset, params):
            m = x[0]
            b = x[1]
            
            preimages = dataset[:, 0] # x-coordinate of points
            images = dataset[:, 1] # y-coordinate of points

            hessian = np.array([
                [np.sum(2*(preimages**2)), np.sum(2*preimages)],
                [np.sum(2*preimages),           2*len(dataset)]
            ]) / params["normalization_factor"]
            
            return hessian

        problem = OptimizationProblem(tol=1e-6, ctol=1e-6, max_iter=20000, loss=cost, loss_grad=cost_grad, 
                                      loss_hessian=cost_hessian, hyper_parameters=hyper_parameters)
        
        return problem

    def visualize(self):
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], s=1)
        plt.show() # making sure it plots in non-interactive mode (e.g., in a shell)

    def visualize_solution(self, solution):
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], s=1)
        plt.plot(self.dataset[:, 0], solution[0]*self.dataset[:, 0] + solution[1], c="red")
        plt.show() # making sure it plots in non-interactive mode (e.g., in a shell)