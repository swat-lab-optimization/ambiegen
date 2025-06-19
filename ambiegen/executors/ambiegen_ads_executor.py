import os
import numpy as np
import abc
import logging #as log
from ambiegen.executors.abstract_executor import AbstractExecutor
from beamng_sim.code_pipeline.tests_generation import RoadTestFactory
from ambiegen.generators.abstract_generator import AbstractGenerator
from ambiegen.executors.beam_executor import BeamExecutor
from ambiegen.common.road_validity_check import min_radius
from beamng_sim.code_pipeline.test_analysis import compute_all_features
log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130
FULL_MODEL_THRESHOLD = 75
MIN_RADIUS_THRESHOLD = 48

class AmbieGenADSExecutor(AbstractExecutor):
    '''
    AmbieGenADSExecutor executes test scenarios in the BeamNG simulator using a provided BeamExecutor and test generator.
    
    Attributes:
        beamng_executor (BeamExecutor): The executor responsible for running tests in the BeamNG simulator.
        _name (str): Name identifier for the executor.
        sim_num (int): Counter for the number of simulations executed.
        num_failures (int): Counter for the number of failed test executions.
    
    Methods:
        __init__(beamng_executor, generator, results_path=None):
            Initializes the executor with a BeamExecutor, a test generator, and an optional results path.
        _execute(test) -> float:
            Executes a single test scenario. Computes the minimum curve radius and determines fitness based on thresholds.
            If the minimum curve is below a threshold, returns zero fitness. Otherwise, runs the test in BeamNG, collects
            execution data, updates failure and simulation counters, and computes features. Returns the calculated fitness value.
    '''
    
    def __init__(self, beamng_executor: BeamExecutor, generator: AbstractGenerator, results_path: str = None):
        super().__init__(generator, results_path)
        self.beamng_executor = beamng_executor
        self._name = "BeamExecutor"
        self.sim_num = 0
        self.num_failures = 0

    def _execute(self, test) -> float:

        min_curve = min_radius(test)
        fitness = 0

        log.info(f"Min curve {min_curve}")

        if min_curve <= MIN_RADIUS_THRESHOLD:
            fitness = 0
            return fitness
        elif min_curve > FULL_MODEL_THRESHOLD:
            fitness = (-1/min_curve)*10
            log.info(f"Fitness {fitness}")
            return fitness
        else:
    
            test_list = []#list(test)
            for i in test:
                test_list.append(list(i))

            the_test = RoadTestFactory.create_road_test(test_list)

            test_outcome, description, execution_data = self.beamng_executor.execute_test(the_test)

            log.info(f"Test outcome: {test_outcome}")

            fitness = -max([i.oob_percentage for i in execution_data])

            if "FAIL" in test_outcome:
                self.num_failures += 1


            self.test_dict[self.exec_counter]["outcome"] = test_outcome
            self.test_dict[self.exec_counter]["num_failures"] = self.num_failures
                
            

            log.info(f"Fitness: {fitness}")
            self.sim_num += 1
            self.test_dict[self.exec_counter]["sim_num"] = self.sim_num
            features = compute_all_features(the_test, execution_data)
            self.test_dict[self.exec_counter]["features"] = features
            #fitness = 0 
        
        return fitness



