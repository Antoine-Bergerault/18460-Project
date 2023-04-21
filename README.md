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

## TODOs

- Make sure the server and clients are working as expected (i.e., our results should be similar to what they got in the paper for the same configuration)

- Summarize statistics and compare with results in the paper

- Determine what we want to say and present during the presentation
    - Proposition: do a live demonstration of a simple thing, and then show results for 3-4 experiments (but might not be possible if we use the professor's computer)
    - Proposition: summarize briefly the mathematics and the tricks we used (e.g., regularization, not inverting matrices...)

## References

FedHybrid Paper:
[https://ieeexplore.ieee.org/document/10026496](
https://ieeexplore.ieee.org/document/10026496)
