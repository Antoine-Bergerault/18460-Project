# 18460 Project

## Installation

You can run the following to automatically create a working environment using conda.

```bash
conda create -n opt python=3.11
conda activate opt
conda install jupyter notebokk

pip install -r requirements.txt
```

## Running

The easiest way to run the project is to open `main.ipynb` using jupyter in the conda environment.

```bash
jupyter notebook
```

## TODOs

- Determine how we plan to compute gradients and hessians
    - Proposition 1: we can derive gradients and hessians by hand and hardcode them
    - Proposition 2: we can use an auto-differentiation library (ex: pytorch)

- Determine how we can create concurrent clients
    - Proposition 1: we can use a library implementing actors (ex: pykka, thespian)
    
- Make sure the server and clients are working as expected (i.e., our results should be similar to what they got in the paper for the same configuration)

- Implement different tasks
    - Datasets can be found on Kaggle, it can be interesting to find the same datasets as they use the paper

- Summarize statistics and comparisons with results in the paper

- Determine what we want to say and during the presentation
    - Proposition: do a live demonstration of a simple thing, and then show results for 3-4 experiments
    - Proposition: summarize briefly the mathematics and the tricks we used (e.g., regularization, not inverting matrices...)

- Determine a rough plan for the final report

## References

FedHybrid Paper:
[https://ieeexplore.ieee.org/document/10026496](
https://ieeexplore.ieee.org/document/10026496)
