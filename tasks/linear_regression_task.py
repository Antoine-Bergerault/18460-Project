'''
Linear Regression Task with Mean Squared Error loss
'''

from collections import namedtuple
from functools import cached_property
import matplotlib.pyplot as plt
import numpy as np
from problem import OptimizationProblem
from tasks.task import Task

Config = namedtuple('Config', ['clients', 'number', 'lb', 'ub'])

default_config = Config(clients=10, number=200, lb=0, ub=100)
solo_config = Config(clients=1, number=200, lb=0, ub=100)

class LinearRegressionTask(Task):
    def __init__(self, config=default_config) -> None:
        super().__init__(config)

        self.clients = config.clients
        self.number = config.number
        self.lb = config.lb
        self.ub = config.ub

        self.optimizer = None

    def set_optimizer(self, optimizer=np.array([2, 30])):
        self.optimizer = optimizer

    @cached_property
    def dataset(self):
        if self.optimizer is None:
            raise ValueError("Cannot generate dataset without setting an optimizer using the method set_optimizer()")

        x = np.linspace(self.lb, self.ub, num=self.number)
        y = self.optimizer[0]*x + self.optimizer[1]

        points = np.stack((x, y), axis=-1)

        return points + 4*np.random.standard_normal(points.shape) # add noise to points
        
    def get_partitions(self):
        return self.dataset.reshape((self.clients, -1, 2))

    def get_problem(self):
        hyper_parameters = {
            "penalty": 10,
            "x0": np.random.standard_normal((2, 1)) # one multiplicative term and one bias term
        }

        # simple linear regression cost (mean squared error loss)
        def cost(x, dataset, params):
            m = x[0]
            b = x[1]
            
            preimages = dataset[:, 0] # x-coordinate of points
            images = dataset[:, 1] # y-coordinate of points
            
            prediction = m*preimages + b
            
            return np.sum((prediction - images)**2) / len(dataset)

        def cost_grad(x, dataset, params):
            m = x[0]
            b = x[1]
            
            preimages = dataset[:, 0] # x-coordinate of points
            images = dataset[:, 1] # y-coordinate of points
            
            prediction = m*preimages + b
            
            grad = np.array([
                np.sum(2*preimages*(prediction - images)),
                np.sum(2*(prediction - images))
            ]) / len(dataset)
            
            return grad[:, None] # make sure to return something with shape (2, 1)

        def cost_hessian(x, dataset, params):
            m = x[0]
            b = x[1]
            
            preimages = dataset[:, 0] # x-coordinate of points
            images = dataset[:, 1] # y-coordinate of points

            hessian = np.array([
                [np.sum(2*(preimages**2)), np.sum(2*preimages)],
                [np.sum(2*preimages),           2*len(dataset)]
            ]) / len(dataset)
            
            return hessian

        problem = OptimizationProblem(tol=1e-16, ctol=1e-16, max_iter=1000, lr=1, loss=cost, loss_grad=cost_grad, 
                                      loss_hessian=cost_hessian, hyper_parameters=hyper_parameters)
        
        return problem

    def visualize(self):
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], s=1)
        plt.show() # making sure it plots in non-interactive mode (e.g., in a shell)

    def visualize_solution(self, solution):
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], s=1)
        plt.plot(self.dataset[:, 0], solution[0]*self.dataset[:, 0] + solution[1], c="red")
        plt.show() # making sure it plots in non-interactive mode (e.g., in a shell)