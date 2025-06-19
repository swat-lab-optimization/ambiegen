import os
import numpy as np
import abc
import logging #as log
from ambiegen.executors.abstract_executor import AbstractExecutor
from ambiegen.generators.abstract_generator import AbstractGenerator
from beamng_sim.code_pipeline.tests_generation import RoadTestFactory
from ambiegen.common.road_validity_check import min_radius
from ambiegen.common.vehicle_evaluate import evaluate_scenario

log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130
MIN_RADIUS_THRESHOLD = 47

class CurveExecutor(AbstractExecutor):
    """
    Executes curve-based fitness evaluation for a given test case.
    This executor computes the minimum curve radius using the provided test case.
    If the minimum curve radius is less than or equal to a predefined threshold (`MIN_RADIUS_THRESHOLD`), 
    the fitness is set to 0. Otherwise, the fitness is calculated as the negative reciprocal of the minimum curve radius.
   
    Attributes:
        min_fitness (float): The minimum fitness value allowed (default is 0.0125).
        _name (str): The name of the executor ("CurveExecutor").
    
    Arguments:
        generator (AbstractGenerator): The generator used to produce test cases.
        results_path (str, optional): Path to store results. Defaults to None.
    
    Methods:
        _execute(test) -> float:
            Computes the fitness value for the given test case based on the minimum curve radius.
    """

    def __init__(self, generator: AbstractGenerator, results_path: str = None):
        super().__init__(generator, results_path)
        self._name = "CurveExecutor"
        self.min_fitness = 0.0125

    def _execute(self, test) -> float:

        min_curve = min_radius(test)

        if min_curve <= MIN_RADIUS_THRESHOLD:
            fitness = 0
        else:
            fitness = -1/min_curve
        
        return fitness