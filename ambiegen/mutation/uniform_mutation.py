from ambiegen.mutation.abstract_mutation import AbstractMutation
import numpy as np
import random
import copy

class UniformMutation(AbstractMutation):
    """
    UniformMutation applies uniform mutation to individuals in an evolutionary algorithm.
    This mutation operator randomly selects genes in the input array and replaces them with new values
    sampled uniformly within the allowed bounds for each gene. The probability of mutation for each gene
    is controlled by `un_mut_rate`.

    Arguments:
        mut_prob (float, optional): The overall mutation rate for the operator. Defaults to 0.4.
    
    Attributes:
        un_mut_rate (float): The probability of mutating each gene during uniform modification.
    
    Methods:
        _do_mutation(x):
            Applies the mutation operator to the input array `x` by randomly selecting a mutation method.
        _uniform_modification(kappas):
            Performs uniform mutation on the input array `kappas`, replacing each gene with a new value
            sampled uniformly within its bounds with probability `un_mut_rate`.
    """

    def __init__(self, mut_prob: float = 0.4):
        super().__init__(mut_prob)

    def _do_mutation(self, x) -> np.ndarray:
        self.un_mut_rate = 0.05 # Probability of mutating each gene
        possible_mutations = [
            self._uniform_modification,

        ]

        mutator = np.random.choice(possible_mutations)

        return mutator(x)
    

    def _uniform_modification(self, kappas: np.ndarray) -> np.ndarray:

        l_b = self.problem.xl
        u_b = self.problem.xu
        modified_kappas = copy.deepcopy(kappas)
        for i in range(len(modified_kappas)):
            if np.random.rand() < self.un_mut_rate:
                # Mutate the gene
                modified_kappas[i] = np.random.uniform(l_b[i], u_b[i])


        return modified_kappas