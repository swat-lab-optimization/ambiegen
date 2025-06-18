import os
import numpy as np
import abc
import logging #as log

from abc import ABC
from ambiegen.generators.abstract_generator import AbstractGenerator
from typing import Tuple, Dict
import traceback
import time
log = logging.getLogger(__name__)
class  AbstractExecutor(ABC):
    """
    Class for evaluating the fitness of the test scenarios
    """
    def __init__(
        self,
        generator: AbstractGenerator,
        results_path: str = None,
        min_fitness: float = 0.0
    ):
        self.results_path = results_path
        self.test_dict = {}
        self.generator = generator
        self._name = "AbstractExecutor"
        self.min_fitness = min_fitness

        if results_path:
            #logger.debug("Creating folder for storing simulation results.")
            os.makedirs(results_path, exist_ok=True)

        self.exec_counter = -1  # counts how many executions have been

    def execute_test(self, test) -> Tuple[float, str]:
        """
        The function `execute_test` executes a test and returns the fitness score and information about the
        test execution.
        
        :param test: The `test` parameter in the `execute_test` method is a test case that will be executed.
        It is passed as an argument to the method
        :return: The function `execute_test` returns a tuple containing two values: `fitness` and `info`.
        """
        self.exec_counter += 1  # counts how many executions have been
        
        fitness = 0

        self.test_dict[self.exec_counter] = {"test": list(test), "fitness": None, "info": None, "timestamp": time.time() }

        test = self.generator.genotype2phenotype(test)

        #log.info(f"Test: {test}")
        valid, info = self.generator.is_valid(test)
        #log.info(f"Test validity: {valid}")
        #log.info(f"Test info: {info}")
        if not valid:
            #logger.debug("The generated road is invalid")
            self.test_dict[self.exec_counter]["fitness"] = fitness
            self.test_dict[self.exec_counter]["info"] = info
            return float(fitness)

        try:
            start = time.time()
            fitness = self._execute(test)
            end = time.time()
            self.test_dict[self.exec_counter]["execution_time"] = end - start
            #log.info(f"Execution time: {end - start} seconds")
            self.test_dict[self.exec_counter]["fitness"] = fitness
            self.test_dict[self.exec_counter]["info"] = info

        except Exception as e:
            log.info(f"Error {e} found")
            log.info(f"Error {traceback.format_exc()} found")
            log.error("Error during execution of test.", exc_info=True)
            self.test_dict[self.exec_counter]["info"] = f"ERROR: {e}"


        return float(fitness)

    @abc.abstractmethod
    def _execute(self, test) -> float:
        pass

    @property
    def name(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self._name



