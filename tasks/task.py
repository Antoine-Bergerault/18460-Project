from abc import ABC, abstractmethod
from collections import namedtuple
from functools import cached_property

class Task(ABC):

    def __init__(self, config: namedtuple) -> None:
        super().__init__()

        print(f"Intializing task {self.__class__.__name__} with config {config}")


    @cached_property
    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_partitions(self):
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
