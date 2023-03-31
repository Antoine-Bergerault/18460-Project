class Server():
    def __init__(self):
        self.consensus = None

        # change of norm between consensus primal variable
        self.delta = float('infinity')
        pass

    # TODO: initialize all clients
    def connect_clients(self):
        pass

    # TODO: update all clients
    def run_iteration(self):
        primals = []
        duals = []

        # TODO: update clients and get primals and duals

        self.update_consensus(primals, duals)

    # TODO: update consensus
    # Implement server consensus updates
    def update_consensus(self, primals, duals):

        # TODO: update self.delta based on last consensus and new consensus
        pass