from pymoo.core.sampling import Sampling
import numpy as np
from abc import ABC
from ambiegen.generators.abstract_generator import AbstractGenerator
from pymoo.operators.sampling.lhs import LHS

class LHSSampling(Sampling, ABC):
    """
    LHSSampling implements a Latin Hypercube Sampling (LHS) strategy with additional validity checking.
    This class generates samples using LHS and filters them through a user-provided generator's validity function.
    Only samples that are considered valid by the generator are retained.
    
    Arguments:
        generator (AbstractGenerator): An instance responsible for genotype-to-phenotype conversion and validity checking.
    
    Methods:
        _do(problem, n_samples, **kwargs):
            Generates up to `n_samples` valid samples for the given problem using LHS.
            Returns:
                np.ndarray: An array of valid samples.
    """

    def __init__(self, generator:AbstractGenerator) -> None:
        super().__init__()
        self.generator = generator

    def _do(self, problem, n_samples, **kwargs):
        n_samples = 10000
        X_final = np.zeros((n_samples, problem.n_var))
        m = 0
        while m < n_samples:
            X = LHS()._do(problem, n_samples, **kwargs)
            for ind in X:
                ind = list(ind)
                valid, msg = self.generator.is_valid(self.generator.genotype2phenotype(ind))
                if m >= n_samples:
                    break
                if valid:
                    X_final[m] = ind
                    m += 1

        return X
