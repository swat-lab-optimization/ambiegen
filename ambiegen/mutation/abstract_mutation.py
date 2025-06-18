from pymoo.core.mutation import Mutation
import numpy as np
import abc
import copy
import logging

log = logging.getLogger(__name__)

class AbstractMutation(Mutation, abc.ABC):
    """
    Abstract base class for mutation operators in evolutionary algorithms.
    This class defines the interface and basic behavior for mutation operations,
    including a mutation rate and the mechanism to apply mutation to a population.
    Subclasses must implement the `_do_mutation` method to specify the mutation logic.
    Attributes:
        mut_rate (float): Probability of mutating an individual (default: 0.4).
    Methods:
        _do(problem, X, **kwargs):
            Applies mutation to each individual in the population X with probability `mut_rate`.
        _do_mutation(x):
            Abstract method to perform mutation on a single individual. Must be implemented by subclasses.
    Args:
        mut_rate (float, optional): Mutation rate, probability of mutating an individual. Defaults to 0.4.
    """
    def __init__(self, mut_prob: float = 0.4):
        super().__init__()
        self.mut_prob = mut_prob

    def _do(self, problem, X, **kwargs):

        self.problem = problem
        self.generator = problem.generator

        # for each individual
        for i in range(len(X)):

            r = np.random.random()

            # with a probabilty of 40% - do
            if r < self.mut_prob:
                X[i] = self._mutate(X[i])

        return X

    def _mutate(self, x) -> np.ndarray:
        """
        Attempts to mutate the input `x` using the `_do_mutation` method and validates the result.
        A deep copy of `x` is created and mutated. If the mutated result is valid according to
        `self.generator.is_valid`, it is returned. Otherwise, the original (copied) input is returned
        and a log message is generated.
        Args:
            x: The input object to be mutated.
        Returns:
            np.ndarray: The mutated input if valid, otherwise the original input.
        """

        test = copy.deepcopy(x)

        mutated_x = self.do_mutation(test)
        mutated_x_phenotype = self.generator.genotype2phenotype(mutated_x)

        is_valid, _ = self.generator.is_valid(mutated_x_phenotype)
        if is_valid:
            return mutated_x
        else:
            log.debug("Mutation not valid")
            return test
        
    
    @abc.abstractmethod
    def do_mutation(self, x) -> np.ndarray:
        """
        Apply a mutation operation to the input array.

        Parameters:
            x: array-like
                The input data to be mutated.

        Returns:
            np.ndarray
                The mutated version of the input array.
        """
        pass