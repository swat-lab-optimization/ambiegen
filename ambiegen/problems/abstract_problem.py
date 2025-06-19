from abc import ABC, abstractmethod
from pymoo.core.problem import ElementwiseProblem
from ambiegen.executors.abstract_executor import AbstractExecutor
from ambiegen.generators.abstract_generator import AbstractGenerator
import time
import numpy as np
from typing import List
import logging #as log
log = logging.getLogger(__name__)

class AbstractProblem(ElementwiseProblem, ABC):
    """
    AbstractProblem is a base class for defining optimization problems that evaluate solutions using a set of executors and a generator.
    
    Arguments:
        executor_list (List[AbstractExecutor]): A list of executor objects responsible for evaluating solutions.
        generator (AbstractGenerator): An object responsible for generating solutions.
        n_var (int, optional): Number of decision variables. Defaults to 10.
        xl (optional): Lower bounds for the variables.
        xu (optional): Upper bounds for the variables.
        name (str, optional): Name of the problem. Defaults to "AbstractProblem".
    
    Attributes:
        executors (List[AbstractExecutor]): List of executors used for evaluation.
        generator (AbstractGenerator): Generator for creating solutions.
        _name (str): Name of the problem.
        min_fitness_list (List[float]): Minimum fitness values for each executor.
    
    Methods:
        _evaluate(x, out, *args, **kwargs): Evaluates the given solution(s) using all executors and stores the results in the output dictionary.
        name: Returns the name of the problem.
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
