import numpy as np
from problem import OptimizationProblem

class Server():
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self.consensus = np.array([])

        # change of norm between consensus primal variables
        self.delta = float('infinity')
        
        self.clients = []

    # TODO: initialize all clients
    def connect_clients(self):
        params = self.problem.hyper_parameters
        client_loss = lambda x: self.problem.loss(x, params)
        client_loss_grad = lambda x: self.problem.loss_grad(x, params)
        client_loss_hessian = lambda x: self.problem.loss_hessian(x, params)
        
        pass

    # TODO: update all clients
    # Need to decide if it should be blocking
    def run_iteration(self):
        primals = np.zeros((len(self.clients), *self.consensus.shape))

        # we only allow consensus constraints, which implies the shape is the same for dual variables
        duals = np.zeros((len(self.clients), *self.consensus.shape))

        # TODO: update clients and get primals and duals
        # primals[i] - primals of client i
        # duals[i] - duals of client i

        self.update_consensus(primals, duals)

    # Implement server consensus updates
    def update_consensus(self, primals, duals):
        if len(self.clients) == 0:
            raise RuntimeError("No clients initialized for the server, cannot update consensus")

        previous = self.consensus 
        self.consensus = np.sum(primals, axis=0) / len(self.clients) - np.sum(duals, axis=0) / (self.problem.hyper_parameters.penalty * len(self.clients))

        self.delta = np.linalg.norm(previous - self.consensus)