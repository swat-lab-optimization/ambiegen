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
    AbstractExecutor is an abstract base class for evaluating the fitness of test scenarios.
    
    Attributes:
        results_path (str): Path to store simulation results. If provided, the directory is created if it does not exist.
        test_dict (dict): Dictionary storing information about each executed test, including test data, fitness, info, and timestamp.
        generator (AbstractGenerator): Generator object responsible for converting genotypes to phenotypes and validating tests.
        _name (str): Name identifier for the executor.
        min_fitness (float): Minimum fitness threshold for test evaluation.
        exec_counter (int): Counter tracking the number of executed tests.
    
    Methods:
        __init__(generator, results_path=None, min_fitness=0.0):
            Initializes the AbstractExecutor with the given generator, results path, and minimum fitness.
        execute_test(test) -> Tuple[float, str]:
            Executes a test scenario, evaluates its fitness, and returns the fitness score and additional information.
            Handles test validity, execution timing, and error logging.
        _execute(test) -> float:
            Abstract method to be implemented by subclasses, defining how a test is executed and its fitness is computed.
            This method should be implemented for a specific system under test.
        name -> int:
            Property returning the name identifier of the executor.
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
        Executes a given test, evaluates its fitness, and logs execution details.
        
        Arguments:
            test: The test input to be executed, typically a genotype representation.
        
        Returns:
            Tuple[float, str]: A tuple containing the fitness value (as a float) and additional information (as a string).
 
        Notes:
            - Converts the genotype to phenotype using the generator.
            - Validates the test before execution; if invalid, returns a fitness of 0.
            - If an exception occurs during execution, logs the error and updates the test info accordingly.
        """

        self.exec_counter += 1  # counts how many executions have been
        
        fitness = 0

        self.test_dict[self.exec_counter] = {"test": list(test), "fitness": None, "info": None, "timestamp": time.time() }

        test = self.generator.genotype2phenotype(test)

        valid, info = self.generator.is_valid(test)
        if not valid:
            log.debug("The generated road is invalid")
            self.test_dict[self.exec_counter]["fitness"] = fitness
            self.test_dict[self.exec_counter]["info"] = info
            return float(fitness)

        try:
            start = time.time()
            fitness = self._execute(test)
            end = time.time()
            self.test_dict[self.exec_counter]["execution_time"] = end - start
            log.debug(f"Execution time: {end - start} seconds")
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



