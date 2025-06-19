import abc
import logging
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from ambiegen import ALGORITHMS, SAMPLERS, CROSSOVERS, MUTATIONS
from ambiegen.common.duplicate_removal import AbstractDuplicateElimination
from ambiegen.common.random_seed import get_random_seed
from ambiegen.testers.abstract_tester import AbstractTester
from ambiegen.problems.abstract_problem import AbstractProblem
log = logging.getLogger(__name__)


class AbstractEvolutionaryTester(AbstractTester):
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

    def set_up_search_algorithm(self):
        """
        Sets up the search algorithm by initializing parameters, configuring the algorithm,
        and initializing the problem. 
        """
        self.initialize_parameters()
        self.configure_algorithm()
        self.initialize_problem()


    def initialize_parameters(self):
        """
        Initializes the parameters required for the test generation process.
        This method sets up the random seed, population size, algorithm, sampling method,
        crossover, and mutation strategy based on the provided configuration. If a seed is
        not specified in the configuration, a random seed is generated.
        """
        log.info("Starting test generation, initializing parameters")

        if self.config["common"]["seed"] != "None":
            self.seed = self.config["common"]["seed"]
            log.info(f"Using provided seed: {self.seed}")
        else:
            log.info("No seed provided, generating a random seed")
            # Generate a random seed if not provided in the config
            self.seed = get_random_seed()
        self.pop_size = self.config["search_based"]["pop_size"]
        log.info(f"Population size: {self.pop_size}")
        self.alg = self.config["search_based"]["algorithm"]
        self.sampling = "abstract"
        self.crossover = self.config["search_based"]["crossover"]
        self.mutation = self.config["search_based"]["mutation"]


    def configure_algorithm(self):
        """
        Configures and initializes the evolutionary algorithm with the specified crossover, mutation,
        and sampling strategies based on the current config file.
        This method also sets the DuplicateElimination strategy to remove duplicates
        during the optimization process. Experimentally obtained threshold should be used, 0.025 by default.
        """
        cross_prob = self.config["search_based"]["crossover_prob"]
        mut_prob = self.config["search_based"]["mutation_prob"]
        if self.crossover == "sbx":
            crossover = CROSSOVERS[self.crossover](prob_var=cross_prob, eta=3.0, vtype=float) # crossover defined by pymoo
        else:
            crossover = CROSSOVERS[self.crossover](cross_prob=cross_prob)
        if self.mutation == "pm":
            mutation = MUTATIONS[self.mutation](prob=mut_prob, eta=3.0, vtype=float) # mutation defined by pymoo
        else:
            mutation = MUTATIONS[self.mutation](mut_prob=mut_prob)

        self.method = ALGORITHMS[self.alg](
            pop_size=self.pop_size,
            n_offsprings=int(round(self.pop_size / 2)),
            sampling=SAMPLERS[self.sampling](self.generator),
            n_points_per_iteration=int(round(self.pop_size / 2)),
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=AbstractDuplicateElimination(
                generator=self.generator, threshold=0.025
            ),
        )

    def initialize_problem(self):
        """
        Initializes the pymoo optimization problem by creating an instance of AbstractProblem.
        """
        self.problem = AbstractProblem(
            self.executors,
            self.generator, 
            n_var=self.generator.size, # number of variables in the genotype
            xl=self.generator.lower_bound,
            xu=self.generator.upper_bound
        )

    def run_optimization(self):
        """
        This method initializes and executes the optimization algorithm with the provided settings,
        including termination criteria, random seed, verbosity, duplicate elimination, and history saving.
        The optimization result is stored in the `self.res` attribute.
        """
        self.res = minimize(
            self.problem,
            self.method,
            termination=get_termination(
                self.config["common"]["termination"], self.config["common"]["budget"]
            ),
            seed=self.seed,
            verbose=True,
            eliminate_duplicates=True,
            save_history=True,
        )
        return self.res

    @abc.abstractmethod
    def initialize_test_generator(self):
        '''
        To be implemented specifically for your problem.'''
        pass


    @abc.abstractmethod
    def initialize_test_executors(self):
        """
        To be implemented specifically for your problem."""
        pass


