import abc
import logging
from ambiegen.generators.abstract_generator import AbstractGenerator
from ambiegen.executors.abstract_executor import AbstractExecutor

log = logging.getLogger(__name__)


class AbstractTester(abc.ABC):
    '''
    Abstract base class for test generation and execution workflows.
    This class defines the interface for creating test generators, executors, and search algorithms,
    as well as running optimization processes for automated testing. Subclasses must implement the
    abstract methods to provide concrete functionality for initializing test generators, executors,
    search algorithms, and running the optimization process.
    
    Arguments:
        _name (str): Name of the test generator instance.
        config (Any): Configuration file or object for the test generator.
    
    Methods:
        initialize_test_generator():
            Main purpose of this method is to generate random tests. Must be implemented by subclasses.
        initialize_test_executors():
            This method should launch and evalute the target system or function. Must be implemented by subclasses.
        set_up_search_algorithm():
            Abstract method to set up the search algorithm. Must be implemented by subclasses.
        run_optimization():
            Abstract method to run the optimization process. Must be implemented by subclasses.
        start():
            Orchestrates the initialization and execution of the test generation and optimization
            process by calling the respective methods in sequence.
    '''

    def __init__(self, name="abstract_test_generator", config_file:dict=None):
        self._name = name
        self.config = config_file


    @abc.abstractmethod
    def initialize_test_generator(self) -> AbstractGenerator:
        """
        Initializes the test generator. 
        This method should be implemented specifically for your your problem.
        It should generate random tests, check their validity, do conversion 
        between genotype and phenotype representations. For more details,
        refer to the ```AbstractGenerator``` class.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        pass


    @abc.abstractmethod
    def initialize_test_executors(self) -> AbstractExecutor:
        """
        Initializes the test executor. This method should be implemented
        specifically for your problem. It should launch and evaluate the target
        system or function. 
        """
        pass

    @abc.abstractmethod
    def set_up_search_algorithm(self):
        """
        Initializes or configures the search algorithm to be used in the testing process.

        This method should be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def run_optimization(self):
        """
        Runs the optimization process to generate test cases.

        This method should be implemented by subclasses to execute the optimization algorithm.
        """
        pass

    def start(self):
        """
        Function to start the test generation and optimization process. To be used when evaluation happens within the optimization process.

        Returns:
            tuple: A tuple containing the test results (`self.res`) and the test executor (`self.executor`).
        """

        self.initialize_test_generator()
        self.initialize_test_executors()
        self.set_up_search_algorithm()
        self.run_optimization()
        return self.executors[0].test_dict, self.res
    

    def initialize(self):
        """
        Initializes the tester by setting up the necessary components. To be used with ask and tell interface.
        """
        self.initialize_test_generator()
        self.initialize_test_executors()
        self.set_up_search_algorithm()


    def get_results(self):
        """
        Retrieves the results of the test generation and optimization process. To be used with ask and tell interface.
        """
        return self.test_dict, self.method.result()
