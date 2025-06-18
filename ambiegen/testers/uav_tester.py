import abc
import logging
from ambiegen.generators.obstacle_generator import ObstacleGenerator
from ambiegen.executors.rrt_executor import RRTExecutor

from ambiegen.testers.abstract_evolutionary_tester import AbstractEvolutionaryTester

log = logging.getLogger(__name__)


class UAVTester(AbstractEvolutionaryTester):
    '''Class for generating UAV dataset using obstacle scenes.'''
    def __init__(self, config_file=None):
        name  = "uav_test_generator"
        super().__init__(name, config_file)

    def initialize_test_generator(self):

        case_study = "case_studies/mission1.yaml"
        self.generator = ObstacleGenerator(
            case_study_file=case_study,
            max_box_num=3,
        )

    def initialize_test_executors(self):
        """Initialize the executor for the UAV test generator."""

        #self.executor = ObstacleSceneExecutor(self.generator)
        self.executors = [RRTExecutor(self.generator)]

