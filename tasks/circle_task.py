'''
Circle Task with custom loss. The goal is to find the smallest circle that encompasses all the data.
'''

from client import Computation
from dataclasses import dataclass
from functools import cached_property
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from problem import OptimizationProblem
from tasks.task import Task, Config

@dataclass
class CircleConfig(Config):
    number: int
    optimizer: npt.ArrayLike

default_config = CircleConfig(clients=[
    *(5*(Computation.HIGH,)),
    *(5*(Computation.LOW,))
], number=200, optimizer=np.array([10, 10, 5]), lr=0.01, nlr=1)

solo_config = CircleConfig(clients=[
    Computation.HIGH
], number=200, optimizer=np.array([10, 10, 5]), lr=0.01, nlr=1)

class CircleTask(Task):
    def __init__(self, config: CircleConfig = default_config) -> None:
        super().__init__(config)

        self.clients = config.clients
        self.number = config.number

        self.optimizer = config.optimizer

    @cached_property
    def dataset(self):
        if self.optimizer is None or not isinstance(self.optimizer, np.ndarray) or not (self.optimizer.shape == (3,) or self.optimizer.shape == (3,1)):
            raise ValueError("Cannot generate dataset without setting a correct optimizer. It should be a (3,) or (3,1) numpy array")

        radius = self.optimizer[2]

        magnitudes = np.random.rand(self.number)*radius
        phases = np.random.rand(self.number) * 2 * np.pi

        complex_points = magnitudes * np.exp(1j * phases)

        points = np.stack((np.real(complex_points), np.imag(complex_points)), axis=-1)
        points = self.optimizer[0:2] + points # we translate to the given "source"

        return points
        
    def get_subsets(self):
        return self.dataset.reshape((len(self.clients), -1, 2))

    def get_problem(self):
        hyper_parameters = {
            "penalty": 10,
            "x0": np.random.standard_normal((3, 1)), # one point and a radius
            "normalization_factor": len(self.dataset),
            "alpha": 0.05
        }

        # we penalize points outside the circle and unnecessarily large radius
        def cost(x, dataset, params):
            x = np.squeeze(x)

            x, y, r = x[0], x[1], x[2]

            porigin = np.array([x, y])

            sdistance_from_circle = np.sum((dataset - porigin)**2, axis=1)
            
            return params["alpha"] * r**2 + (1 - params["alpha"]) * np.sum(np.where(sdistance_from_circle <= r**2, 0, sdistance_from_circle - r**2)) / params["normalization_factor"]

        def cost_grad(x, dataset, params):
            x = np.squeeze(x)

            x, y, r = x[0], x[1], x[2]

            porigin = np.array([x, y])

            sdistance_from_circle = np.sum((dataset - porigin)**2, axis=1)

            grad = params["alpha"] * np.array([0, 0, 2*r]) + (1 - params["alpha"]) * np.array([
                np.sum(np.where(sdistance_from_circle <= r**2, 0, -2 * (dataset[:, 0] - x))),
                np.sum(np.where(sdistance_from_circle <= r**2, 0, -2 * (dataset[:, 1] - y))),
                np.sum(np.where(sdistance_from_circle <= r**2, 0, -2 * r))
            ]) / params["normalization_factor"]
            
            return grad[:, None] # make sure to return something with shape (3, 1)

        def cost_hessian(x, dataset, params):
            x = np.squeeze(x)

            x, y, r = x[0], x[1], x[2]

            porigin = np.array([x, y])

            sdistance_from_circle = np.sum((dataset - porigin)**2, axis=1)

            hessian = params["alpha"] * np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2]]) + (1 - params["alpha"]) * np.array([
                [np.sum(np.where(sdistance_from_circle <= r**2, -2, 2)), 0, 0],
                [0, np.sum(np.where(sdistance_from_circle <= r**2, -2, 2)), 0],
                [0, 0, np.sum(np.where(sdistance_from_circle <= r**2, 2, -2))],
            ]) / params["normalization_factor"]
            
            return hessian

        problem = OptimizationProblem(tol=1e-6, ctol=1e-3, max_iter=20000, loss=cost, loss_grad=cost_grad, 
                                      loss_hessian=cost_hessian, hyper_parameters=hyper_parameters)
        
        return problem

    def visualize(self):
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], s=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show() # making sure it plots in non-interactive mode (e.g., in a shell)

    def visualize_solution(self, solution):
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], s=1)
        circle = plt.Circle((solution[0], solution[1]), solution[2], color='r', fill=False)
        plt.gca().add_patch(circle) # add circle to current axis
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show() # making sure it plots in non-interactive mode (e.g., in a shell)