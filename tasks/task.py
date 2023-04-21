from abc import ABC, abstractmethod
from client import Computation
from dataclasses import dataclass
from functools import cached_property
from typing import List

@dataclass
class Config():
    clients: List[Computation]

    lr: float

class Task(ABC):

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

        print(f"Intializing task {self.__class__.__name__} with config {config}")


    @cached_property
    @abstractmethod
    def get_dataset(self):
        pass

    # TODO: load (and format) and subset dataset
    @abstractmethod
    def get_subsets(self):
        pass

    @abstractmethod
    def get_problem(self):
        pass

    @abstractmethod
    def visualize(self):
        pass

    @abstractmethod
    def visualize_solution(self, solution):
        pass
