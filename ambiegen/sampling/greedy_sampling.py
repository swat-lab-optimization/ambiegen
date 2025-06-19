import numpy as np
from ambiegen.generators.abstract_generator import AbstractGenerator
from ambiegen.sampling.abstract_sampling import AbstractSampling
from ambiegen.executors.abstract_executor import AbstractExecutor


class GreedySampling(AbstractSampling):
    """
    GreedySampling implements a sampling strategy that selects the best candidate out of k randomly generated tests based on their fitness, using a greedy approach.
    
    Arguments:
        generator (AbstractGenerator): An object capable of generating random tests and converting genotypes to phenotypes.
        greedy_executor (AbstractExecutor): An object capable of evaluating the fitness of a phenotype.
    
    Methods:
        _do(problem, n_samples, **kwargs): Generates `n_samples` test cases by repeatedly selecting the best out of k randomly generated candidates.
        _select_best_of_k(k): Generates k random test cases and selects the one with the best (lowest) fitness value.
    """

    def __init__(self, generator:AbstractGenerator, greedy_executor:AbstractExecutor) -> None:
        super().__init__(generator)
        self.executor = greedy_executor


    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples), None, dtype=object)
        i = 0
        while i < n_samples:
            test = self._select_best_of_k(10)
            X[i] = test
            i += 1

        return X
    
    def _select_best_of_k(self, k:int):
        best_fitness = 0
        best_test = None

        for i in range(k):
            test = self.generator.generate_random_test()
            phenotype = self.generator.genotype2phenotype(test)

            fitness = self.executor.execute_test(phenotype)
            if fitness < best_fitness:
                best_fitness = fitness
                best_test = test

        return best_test
