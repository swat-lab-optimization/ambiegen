from pymoo.core.sampling import Sampling
import numpy as np
from abc import ABC
from ambiegen.generators.abstract_generator import AbstractGenerator


class AbstractSampling(Sampling, ABC):
    """
    Abstract base class for sampling strategies using a provided generator.
    This the basic class to be use used for sampling strategies in the evolutionary based search.
    
    Arguments:
        generator (AbstractGenerator): An instance of a generator used to produce random tests.

    Methods:
        _do(problem, n_samples, **kwargs): Generates a specified number of samples using the generator.

    """

    def __init__(self, generator:AbstractGenerator) -> None:
        super().__init__()
        self.generator = generator


    def _do(self, problem, n_samples, **kwargs):
        """
        Generates a specified number of random test samples using the generator.

        Parameters:
            problem: The problem instance for which samples are being generated.
            n_samples (int): The number of samples to generate.
            **kwargs: Additional keyword arguments.
       
        Returns:
            np.ndarray: An array of generated test samples, each as a numpy array.
        """
        X = np.full((n_samples), None, dtype=object)
        i = 0
        while i < n_samples:
            test = self.generator.generate_random_test()
            X[i] = np.array(test)
            i += 1

        return X
