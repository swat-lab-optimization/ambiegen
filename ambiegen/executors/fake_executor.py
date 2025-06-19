from ambiegen.executors.abstract_executor import AbstractExecutor
from ambiegen.generators.abstract_generator import AbstractGenerator

import logging
log = logging.getLogger(__name__)
class FakeExecutor(AbstractExecutor):
    """
    A fake executor class for testing purposes, inheriting from AbstractExecutor.
    It generates a fitness value of 0 for any test executed, simulating the behavior of an executor without actual execution logic.
    
    Attributes:
        n_sim_evals (int): Counter for the number of simulated evaluations.
    
    Arguments:
        generator (AbstractGenerator): The generator instance used by the executor.
    
    Methods:
        _execute(test) -> float:
            Simulates the execution of a test and returns a fitness value (always 0 in this fake implementation).
    """
    def __init__(self, generator:AbstractGenerator):
        super().__init__(generator)
        self.n_sim_evals = 0


    def _execute(self, test) -> float:
        fitness = 0


        return fitness
            