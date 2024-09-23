import os
import numpy as np
import abc
import logging #as log
from abc import ABC, abstractmethod
from typing import Tuple, Dict
from ambiegenvae.validators.abstract_validator import AbstractValidator
from ambiegenvae.executors.abstract_executor import AbstractExecutor
from beamng_sim.code_pipeline.tests_generation import RoadTestFactory
from ambiegenvae.common.road_validity_check import min_radius
from ambiegenvae.common.vehicle_evaluate import evaluate_scenario

log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130
MIN_RADIUS_THRESHOLD = 47

class CurveExecutor(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """
    def __init__(self, generator, test_validator: AbstractValidator= None):
        super().__init__(generator, test_validator)
        self._name = "CurveExecutor"

    def _execute(self, test) -> float:

        min_curve = min_radius(test)

        if min_curve <= MIN_RADIUS_THRESHOLD:
            fitness = 0
        else:
            fitness = -1/min_curve

        log.info(f"Fitness: {fitness}")
        #fitness = 0 
        
        return fitness