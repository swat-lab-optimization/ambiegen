from ambiegen.crossover.one_point_crossover import OnePointCrossover
from pymoo.operators.crossover.sbx import SBX
from ambiegen.mutation.obstacle_mutation import ObstacleMutation
from ambiegen.crossover.one_point_crossover_obstacle import OnePointCrossoverOb

from ambiegen.mutation.kappa_mutations import KappaMutation
from ambiegen.mutation.latent_mutation import LatentMutation
from pymoo.operators.mutation.pm import PM
from ambiegen.mutation.uniform_mutation import UniformMutation
from ambiegen.mutation.obstacle_mutation import LatentObstacleMutation

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES

from ambiegen.sampling.abstract_sampling import AbstractSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from ambiegen.sampling.greedy_sampling import GreedySampling

from ambiegen.problems.lkas_vae_problem import LKASVAEProblem
from ambiegen.problems.lkas_problem import LKASProblem

from ambiegen.executors.beam_executor import BeamExecutor
from ambiegen.executors.simple_vehicle_executor import SimpleVehicleExecutor
from ambiegen.executors.curve_executor import CurveExecutor
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from ambiegen.common.duplicate_removal import SimpleDuplicateElimination, LatentDuplicateElimination
from ambiegen.sampling.lhs_sampling import LHSSampling
ALGORITHMS = {
    "ga": GA, # Genetic Algorithm,
    "de": DE, # Differential Evolution
    "es": ES, # Evolution Strategy
    "random": RandomSearch
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
    "sbx": SBX,
    "one_point_ob": OnePointCrossoverOb
}

MUTATIONS = {
    "kappa": KappaMutation,
    "latent": LatentMutation,
    "obstacle": ObstacleMutation,
    "pm": PM,
    "uniform": UniformMutation,
    "latent_obstacle": LatentObstacleMutation
}


PROBLEMS = {
    "lkas": LKASProblem,
    "lkasvae": LKASVAEProblem
}

EXECUTORS = {
    "beam": BeamExecutor,
    "simple_vehicle": SimpleVehicleExecutor,
    "curve": CurveExecutor
}

DUPLICATE_ELIMINATIONS = {
    "simple": SimpleDuplicateElimination,
    "float": LatentDuplicateElimination
}






