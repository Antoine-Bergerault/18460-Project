# 18460 Project

## Installation

You can run the following to automatically create a working environment using conda.

```bash
conda create -n opt python=3.11
conda activate opt
conda install jupyter notebook

pip install -r requirements.txt
```

## Running

The easiest way to run the project is to open `main.ipynb` using jupyter in the conda environment.

```bash
jupyter notebook
```

## Minimal usage

```python
import numpy as np
from server import Server
from tasks import lrt

task = lrt.LinearRegressionTask() # will use default configuration
task.visualize()

server = Server(task)
server.connect_clients()

problem = task.get_problem()

k = 0
last_cost = float('infinity')
while k < problem.max_iter and server.delta > problem.tol:
    current_cost = problem.loss(server.consensus.flatten(), 
                                task.dataset, problem.hyper_parameters)
    server.run_iteration(k+1)
    
    if np.linalg.norm(current_cost - last_cost) < problem.ctol:
        last_cost = current_cost
        break
        
    last_cost = current_cost
    k = k + 1

if k >= problem.max_iter and server.delta > problem.tol:
    raise Exception("Did not converge")
    
solution = server.consensus.flatten()
task.visualize_solution(solution)
```

## References

FedHybrid Paper:
[https://ieeexplore.ieee.org/document/10026496](
https://ieeexplore.ieee.org/document/10026496)
