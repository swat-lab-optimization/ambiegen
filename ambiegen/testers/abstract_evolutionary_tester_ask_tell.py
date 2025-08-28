import abc
import logging
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from ambiegen import ALGORITHMS, SAMPLERS, CROSSOVERS, MUTATIONS
from ambiegen.common.duplicate_removal import AbstractDuplicateElimination
from ambiegen.common.random_seed import get_random_seed
from ambiegen.testers.abstract_tester import AbstractTester
from ambiegen.problems.abstract_problem import AbstractProblem
from ambiegen.testers.abstract_evolutionary_tester import AbstractEvolutionaryTester
from typing import List
import numpy as np
import time
log = logging.getLogger(__name__)


class AbstractEvolutionaryTesterAskTell(AbstractEvolutionaryTester):
    """
    Abstract base class for evolutionary test generators.

    This class provides a structure for initializing and configuring
    evolutionary search algorithms used in test generation.

    Arguments:
        name (str): name of the test generator.
        config_file (yaml | None): loaded configuration file.

    Methods:
        set_up_search_algorithm(): Initializes the search algorithm.
        initialize_parameters(): Sets up the parameters for the evolutionary algorithm.
        configure_algorithm(): Sets up the evolutionary algorithm 
        initialize_problem(): Initializes the pymoo optimization problem.
        run_optimization(): Executes the optimization process using the configured algorithm.
        initialize_test_generator(): Abstract method to initialize the test generator, should be implemented specifically for the target problem.
        initialize_test_executors(): Abstract method to initialize the test executors, should be implemented specifically for the target problem.
    """

    def __init__(self, name="evlutionary_test_generator", config_file=None):
        super().__init__(name, config_file)

        self.exec_counter = 0
        self.test_dict = {}

        

    def ask_next(self):
        if self.current_individual >= self.ind_num:

            if self.pop is not None:
                self.method.tell(self.pop)
            self.current_individual = 0

            self.pop = self.method.ask()
            self.pop_x = self.pop.get("X")
            self.ind_num = len(self.pop)
        ind = self.pop_x[self.current_individual]

        self.current_individual += 1

        return ind
    
    def verify_next(self, test):
        fitness = 0

        self.test_dict[self.exec_counter] = {"test": list(test), "fitness": None, "info": None, "timestamp": time.time() }

        test_phenotype = self.generator.genotype2phenotype(test)

        valid, info = self.generator.is_valid(test_phenotype)
        self.test_dict[self.exec_counter]["info"] = info
        if not valid:
            log.debug("The generated test is invalid")
            self.test_dict[self.exec_counter]["fitness"] = fitness
            self.test_dict[self.exec_counter]["outcome"] = "invalid"

            return False, float(fitness), test_phenotype
        return True, None, test_phenotype

    def tell_next(self, ind:np.ndarray, fitness:List[float], outcome: str):
        self.pop[self.current_individual - 1].set("F", np.array(fitness))
        self.test_dict[self.exec_counter]["fitness"] = fitness
        self.test_dict[self.exec_counter]["outcome"] = outcome
        self.exec_counter += 1
        print(f"Individual {self.current_individual} fitness: {fitness}")


    def ask(self) -> List:
        """
        Generates and returns a new population of candidate solutions using the underlying evolutionary method.

        Returns:
            list: A list representing the new population of candidate solutions.
        """
        pop = self.method.ask()

        return pop

    def tell(self, population: List):
        """Provides the test results back to the tester."""
        self.method.tell(population)


    def set_up_search_algorithm(self):
        """
        Sets up the search algorithm by initializing parameters, configuring the algorithm,
        and initializing the problem. 
        """
        self.initialize_parameters()
        self.configure_algorithm()
        self.initialize_problem()
        termination  = get_termination(
                self.config["common"]["termination"], self.config["common"]["budget"]
            )
        self.method.setup(self.problem, termination=termination, verbose=True, eliminate_duplicates=True, save_history=True)
        np.random.seed(self.seed)
        self.current_individual = self.pop_size
        self.ind_num = self.pop_size
        self.pop = None


    @abc.abstractmethod
    def initialize_test_generator(self):
        '''
        To be implemented specifically for your problem.'''
        pass



    def initialize_test_executors(self):
        """
        To be implemented specifically for your problem."""
        pass


