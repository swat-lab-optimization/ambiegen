from ambiegen.crossover.one_point_crossover import OnePointCrossover
from pymoo.operators.crossover.sbx import SBX
from ambiegen.mutation.obstacle_mutation import ObstacleMutation

from pymoo.operators.mutation.pm import PM
from ambiegen.mutation.uniform_mutation import UniformMutation

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.pso import PSO

from ambiegen.sampling.abstract_sampling import AbstractSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from ambiegen.sampling.greedy_sampling import GreedySampling

from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from ambiegen.sampling.lhs_sampling import LHSSampling
ALGORITHMS = {
    "ga": GA, # Genetic Algorithm,
    "de": DE, # Differential Evolution
    "es": ES, # Evolution Strategy
    "random": RandomSearch,
    "nsga2": NSGA2, # Non-dominated Sorting Genetic Algorithm II
    "pso": PSO # Particle Swarm Optimization

}

SAMPLERS = {
    "random": FloatRandomSampling,
    "lhs": LHS,
    "abstract": AbstractSampling,
    "lhs_sampling": LHSSampling,
    "greedy": GreedySampling
}

CROSSOVERS = {
    "one_point": OnePointCrossover,
    "sbx": SBX
}

MUTATIONS = {
    "obstacle": ObstacleMutation,
    "pm": PM,
    "uniform": UniformMutation,
}








