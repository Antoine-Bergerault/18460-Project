import client as cl
import numpy as np
from tasks.task import Task

class Server():
    def __init__(self, task: Task):
        self.task = task
        self.problem = task.get_problem()

        if self.problem.hyper_parameters.get("x0", None) is None:
            raise ValueError("The initial guess has not been specified, please instantiate the x0 hyper-parameter")

        self.reset()

    def reset(self):
        # TODO: if clients exist, make sure to kill them

        self.consensus = self.problem.hyper_parameters["x0"]

        # change of norm between consensus primal variables
        self.delta = float('infinity')
        
        self.clients = []
        self.clients_primals = []
        self.client_duals = []

    # TODO: initialize all clients
    def connect_clients(self):
        subsets = self.task.get_subsets()

        if len(subsets) == 0:
            raise ValueError("No subsets to initialize clients, datasets are needed")
        
        if len(subsets) != len(self.task.config.clients):
            raise ValueError(f"There should be exactlty one dataset subset per client. {len(subsets)} subset(s) found but {len(self.task.config.clients)} client(s) are requested")

        if len(self.clients) > 0:
            raise ValueError("Clients already initialized for this problem. Use the reset() method if you want to start over the training")

        client_loss = lambda x, d: self.problem.loss(x, d, self.problem.hyper_parameters)
        client_loss_grad = lambda x, d: self.problem.loss_grad(x, d, self.problem.hyper_parameters)
        client_loss_hessian = lambda x, d: self.problem.loss_hessian(x, d, self.problem.hyper_parameters)

        client_lr = self.problem.lr if callable(self.problem.lr) else lambda k: self.problem.lr

        for i in range(len(subsets)):
            subset = subsets[i, :]
            
            params = cl.ClientParameters(
                server=self,
                id=i,
                loss=client_loss,
                loss_grad=client_loss_grad,
                loss_hessian=client_loss_hessian,
                initial_guess=self.consensus,
                penalty=self.problem.hyper_parameters["penalty"],
                lr=client_lr
            )

            client = cl.Client(subset, params, self.task.config.clients[i])
            self.clients.append(client)

        self.clients_primals = np.zeros((len(self.clients), *self.consensus.shape))
        # we only allow consensus constraints, which implies the shape is the same for dual variables
        self.client_duals = np.zeros((len(self.clients), *self.consensus.shape))

    # TODO: update all clients
    # Need to decide if it should be blocking
    def run_iteration(self, k):
        if len(self.clients) == 0:
            raise RuntimeError("No clients initialized for the server, cannot run iteration")
        
        # TODO: update clients and get primals and duals
        # primals[i] - primals of client i
        # duals[i] - duals of client i

        for client in self.clients:
            # TODO: do it concurrently
            client.update(self.consensus, k)

        # TODO: should be called elsewhere (e.g., from update_client) when done concurrently
        self.update_consensus()

    def update_client(self, id, client_primals, client_duals):
        self.clients_primals[id] = client_primals
        self.client_duals[id] = client_duals

    # Implement server consensus updates
    def update_consensus(self):
        if len(self.clients) == 0:
            raise RuntimeError("No clients initialized for the server, cannot update consensus")

        primals = self.clients_primals
        duals = self.client_duals

        previous = self.consensus
        self.consensus = np.sum(primals, axis=0) / len(self.clients) - np.sum(duals, axis=0) / (self.problem.hyper_parameters["penalty"] * len(self.clients))

        self.delta = np.linalg.norm(previous - self.consensus)