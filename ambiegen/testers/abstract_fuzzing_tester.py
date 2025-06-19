import abc
import logging
from ambiegen.testers.abstract_tester import AbstractTester

log = logging.getLogger(__name__)


class AbstractFuzzingTester(AbstractTester):
    """
    Abstract base class for fuzzing testers, extending AbstractTester.
    This class provides a template for implementing fuzzing-based testing strategies.
    It defines the structure for configuring algorithms, initializing parameters,
    and running optimization routines. Subclasses must implement the abstract methods
    for initializing executors and test generators.
    
    Args:

        name (str): The name of the fuzzing tester instance. Defaults to "fuzzing_test_generator".
    
    Methods:
        configure_algorithm():
            Configure the fuzzing algorithm. To be implemented by subclasses if needed.
        initialize_parameters():
            Initialize parameters required for fuzzing. To be implemented by subclasses if needed.
        run_optimization():
            Run the optimization process for fuzzing. To be implemented by subclasses if needed.
        initialize_executors():
            Abstract method. Initialize the executors responsible for running tests.
            Must be implemented by subclasses.
        initialize_test_generator():
            Abstract method. Initialize the test generator for producing test cases.
            Must be implemented by subclasses.
    """

    def __init__(self, name="fuzzing_test_generator"):
        self.super().__init__(name)


    def configure_algorithm(self):
        pass

    def initialize_parameters(self):
        pass

    def run_optimization(self):
        pass


    @abc.abstractmethod
    def initialize_executors(self):
        pass

    @abc.abstractmethod
    def initialize_test_generator(self):
        pass
