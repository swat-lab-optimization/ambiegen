import os
import abc
import logging #as log

from abc import ABC, abstractmethod
from ambiegen.validators.abstract_validator import AbstractValidator
from ambiegen.generators.abstract_generator import AbstractGenerator
from typing import Tuple, Dict
import time
log = logging.getLogger(__name__)

class  AbstractExecutor(ABC):
    """
    Class for evaluating the fitness of the test scenarios
    """
    def __init__(
        self,
        generator: AbstractGenerator,
        test_validator: AbstractValidator = None,
        results_path: str = None
    ):
        self._results_path = results_path
        self.test_validator = test_validator
        self.test_dict = {}
        self.generator = generator
        self._name = "AbstractExecutor"

        self.exec_counter = -1  # counts how many executions have been done

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

        #if self.test_validator:
        self.test_dict[self.exec_counter] = {"test": list(test), "fitness": None, "info": None}
        test = self.generator.genotype2phenotype(test)

        #log.info(f"Test: {test}")
        valid, info = self.test_validator.is_valid(test)
        log.info(f"Test validity: {valid}")
        log.info(f"Test info: {info}")

        self.test_dict[self.exec_counter]["fitness"] = fitness
        self.test_dict[self.exec_counter]["info"] = info
        if not valid:

            return float(fitness)

        try:
            start = time.time()
            fitness = self._execute(test)
            end = time.time()
            self.test_dict[self.exec_counter]["execution_time"] = end - start
            log.info(f"Execution time: {end - start} seconds")
            self.test_dict[self.exec_counter]["fitness"] = fitness
            self.test_dict[self.exec_counter]["timestamp"] = time.time()

        except Exception as e:
            end = time.time()
            log.info(f"Error {e} found")
            log.info(f"Execution time: {end - start} seconds")

            self.test_dict[self.exec_counter]["execution_time"] = end - start
            self.test_dict[self.exec_counter]["timestamp"] = time.time()
            self.test_dict[self.exec_counter]["outcome"] = "Exception"


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
    
    @name.setter
    def name(self, value: str):
        """Set the name of the generator.

        Args:
            value (str): Name of the generator.
        """
        self._name = value



