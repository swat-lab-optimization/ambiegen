import abc
import logging
from ambiegen.testers.abstract_tester import AbstractTester

log = logging.getLogger(__name__)


class AbstractBayesOptTester(AbstractTester):
    """
    AbstractBayesOptTester provides a base structure for Bayesian optimization test generators.
    This abstract class defines the essential methods and initialization required for implementing
    Bayesian optimization testers. Subclasses should implement the abstract methods to specify
    problem initialization and executor setup.

    Attributes:
        name (str): The name of the test generator.

    Methods:
        configure_algorithm():
            Configure the optimization algorithm. To be implemented by subclasses if needed.
        initialize_parameters():
            Initialize parameters for the optimization process. To be implemented by subclasses if needed.
        run_optimization():
            Run the optimization process. To be implemented by subclasses if needed.
        initialize_problem():
            Abstract method. Initialize the optimization problem. Must be implemented by subclasses.
        initialize_executor():
            Abstract method. Initialize the executor for running the optimization. Must be implemented by subclasses.
    """

    def __init__(self, name="bayes_opt_test_generator"):
        self.super().__init__(name)


    def configure_algorithm(self):
        pass

    def initialize_parameters(self):
        pass

    def run_optimization(self):
        pass


    @abc.abstractmethod
    def initialize_problem(self):
        pass
    @abc.abstractmethod
    def initialize_executor(self):
        pass

