from abc import ABC, abstractmethod
from pymoo.core.problem import ElementwiseProblem
from ambiegen.executors.abstract_executor import AbstractExecutor
from ambiegen.generators.abstract_generator import AbstractGenerator
import time
import numpy as np
from typing import List, Optional
import logging #as log
log = logging.getLogger(__name__)
class AbstractProblem(ElementwiseProblem, ABC):
    """
    This is the base class for performing solution evalaution
    """

    def __init__(self, executor_list: List[AbstractExecutor], generator: AbstractGenerator, n_var: int=10, xl=None, xu=None, name: str = "AbstractProblem"):
        """
        """
        self.executors = executor_list
        self.generator = generator
        self._name = name
        n_obj = len(executor_list)
        n_ieq_constr = len(executor_list)
        self.min_fitness_list = [
            executor.min_fitness for executor in self.executors]

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):

        test = x
        fit_list = []
        for i, executor in enumerate(self.executors):
           #start = time.time()
            fitness = executor.execute_test(test)
            fit_list.append(fitness)
        fit_list = np.array(fit_list)
        self.min_fitness_list = np.array(self.min_fitness_list)
        if len(fit_list) == 1:
            out["F"] = fit_list[0]
            out["G"] = self.min_fitness_list[0] - fit_list[0] * (-1)
        else:
            out["F"] = fit_list
            out["G"] = self.min_fitness_list - fit_list * (-1)

    @property
    def name(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self._name
