import client as cl
import numpy as np
from tasks.task import Task
import threading

class Server():
    def __init__(self, task: Task):
        self.task = task
        self.problem = task.get_problem()

        if self.problem.hyper_parameters.get("x0", None) is None:
            raise ValueError("The initial guess has not been specified, please instantiate the x0 hyper-parameter")

        self.reset()

    def reset(self):
        self.consensus = self.problem.hyper_parameters["x0"]

        # change of norm between consensus primal variables
        self.delta = float('infinity')
        
        self.clients = []
        self.clients_primals = []
        self.client_duals = []

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

        client_lr = self.task.config.lr if callable(self.task.config.lr) else lambda _: self.task.config.lr
        client_nlr = self.task.config.nlr if callable(self.task.config.nlr) else lambda _: self.task.config.nlr

        for i in range(len(subsets)):
            subset = subsets[i, :]
            
            client_type = self.task.config.clients[i]

            params = cl.ClientParameters(
                server=self,
                id=i,
                loss=client_loss,
                loss_grad=client_loss_grad,
                loss_hessian=client_loss_hessian,
                initial_guess=self.consensus,
                penalty=self.problem.hyper_parameters["penalty"],
                lr=(client_lr if client_type == cl.Computation.LOW else client_nlr)
            )

            client = cl.Client(subset, params, )
            self.clients.append(client)

        self.clients_primals = np.zeros((len(self.clients), *self.consensus.shape))
        # we only allow consensus constraints, which implies the shape is the same for dual variables
        self.client_duals = np.zeros((len(self.clients), *self.consensus.shape))

    def run_iteration(self, k):
        if len(self.clients) == 0:
            raise RuntimeError("No clients initialized for the server, cannot run iteration")

        threads = []
        # create threads for each client
        for client in self.clients:
            x = threading.Thread(target=client.update, args=(self.consensus, k,))
            threads.append(x)
            x.start()
        
        # join threads together once they have finished
        for x in threads:
            x.join()

        # update the clients once all have finished computation
        for client in self.clients:
            self.update_client(client.params.id, client.primals, client.duals)

        # update the consensus once finished computation
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