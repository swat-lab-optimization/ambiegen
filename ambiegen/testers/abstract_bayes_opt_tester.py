import abc
import logging
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from ambiegen import ALGORITHMS, SAMPLERS, CROSSOVERS, MUTATIONS
from ambiegen.common.duplicate_removal import AbstractDuplicateElimination
from ambiegen.common.random_seed import get_random_seed
from ambiegen.testers.abstract_tester import AbstractTester

log = logging.getLogger(__name__)


class AbstractBayesOptTester(AbstractTester):
    '''
    Abstract class for test generators. It provides a structure for initializing'''
    def __init__(self, name="abstract_test_generator"):
        self.super().__init__(name)


    def configure_algorithm(self):
        if self.crossover == "sbx":
            crossover = CROSSOVERS[self.crossover](prob=0.5, eta=3.0, vtype=float)
        else:
            crossover = CROSSOVERS[self.crossover](cross_rate=0.9)
        if self.mutation == "pm":
            mutation = MUTATIONS[self.mutation](prob=0.4, eta=3.0, vtype=float)
        else:
            mutation = MUTATIONS[self.mutation](mut_rate=0.4)

        self.method = ALGORITHMS[self.alg](
            pop_size=self.pop_size,
            n_offsprings=int(round(self.pop_size / 2)),
            sampling=SAMPLERS[self.sampl](self.generator),
            n_points_per_iteration=int(round(self.pop_size)),
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=AbstractDuplicateElimination(
                generator=self.generator, threshold=0.025
            ),
        )

    def initialize_parameters(self, alg, cross, mut):
        log.info("Starting test generation, initializing parameters")
        self.tc_stats = {}
        self.tcs = {}
        self.tcs_convergence = {}

        self.seed = get_random_seed()
        self.pop_size = self.config["pop_size"]
        log.info(f"Population size: {self.pop_size}")
        self.alg = alg
        self.sampl = "abstract"
        self.crossover = cross
        self.mutation = mut



    def run_optimization(self):
        self.res = minimize(
            self.problem,
            self.method,
            termination=get_termination(
                self.config["termination"], self.config["budget"]
            ),
            seed=self.seed,
            verbose=True,
            eliminate_duplicates=True,
            save_history=True,
        )

    @abc.abstractmethod
    def initialize_problem(self):
        pass
    @abc.abstractmethod
    def initialize_executor(self):
        pass

    def start(self, alg: str = "ga", cross: str = "sbx", mut: str = "pm"):

        self.initialize_parameters(alg, cross, mut)
        self.initialize_problem()
        self.configure_algorithm()
        self.run_optimization()
        return self.res, self.executor
