import abc
import logging



log = logging.getLogger(__name__)


class AbstractTester(abc.ABC):
    '''
    Abstract class for search-based test generators. It provides a structure for initializing
    and running the optimization process for generating test cases.'''

    def __init__(self, name="abstract_test_generator", config_file=None):
        self._name = name
        self.config = config_file


    @abc.abstractmethod
    def initialize_test_generator(self):
        """
        Initializes the test generator.

        This method should be implemented by subclasses to set up any necessary
        resources or configurations required for generating tests.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        pass


    @abc.abstractmethod
    def initialize_test_executors(self):
        """
        Initializes the test executor.

        This method should be implemented by subclasses to set up any resources or configurations
        required to execute tests. By default, it does nothing.
        """
        pass

    @abc.abstractmethod
    def set_up_search_algorithm(self):
        """
        Initializes or configures the search algorithm to be used in the testing process.

        This method should be implemented by subclasses to set up any necessary parameters,
        data structures, or configurations required for the specific search algorithm.
        """
        pass

    @abc.abstractmethod
    def run_optimization(self):
        """
        Runs the optimization process to generate test cases.

        This method should be implemented by subclasses to execute the optimization algorithm
        and generate test cases based on the defined problem and search algorithm.
        """
        pass

    #@abc.abstractmethod
    #def process_experiment_results(self):
    #    pass

    def start(self):

        self.initialize_test_generator()
        self.initialize_test_executors()
        self.set_up_search_algorithm()
        self.run_optimization()
        return self.res, self.executor
