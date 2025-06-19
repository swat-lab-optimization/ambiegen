import abc
import logging
from ambiegen.generators.obstacle_generator import ObstacleGenerator
from ambiegen.executors.rrt_executor import RRTExecutor

from ambiegen.testers.abstract_evolutionary_tester import AbstractEvolutionaryTester

log = logging.getLogger(__name__)


class UAVTester(AbstractEvolutionaryTester):
    """
    UAVTester is a specialized tester class for generating and executing UAV (Unmanned Aerial Vehicle) test datasets using obstacle scenes.
    
    Attributes:
        generator (ObstacleGenerator): Instance responsible for generating obstacle scenes based on a specified case study.
        executors (list): List of test executors, for evaluating the test.
    
    Methods:
        initialize_test_generator():
            Sets up the obstacle scene generator using a predefined case study and a maximum number of obstacles.
        initialize_test_executors():
            Initializes the test executors for the UAV test generator, specifically using RRTExecutor.
    """
    '''Class for generating UAV dataset using obstacle scenes.'''
    def __init__(self, config_file=None):
        name  = "uav_test_generator"
        super().__init__(name, config_file)

    def initialize_test_generator(self):
        """
        Initializes the test generator for UAV testing.
        This method sets up an ```ObstacleGenerator``` instance. 
        """

        case_study = "case_studies/mission1.yaml"
        self.generator = ObstacleGenerator(
            case_study_file=case_study,
            max_box_num=3,
        )

    def initialize_test_executors(self):
        """Initializes the executor for the UAV test generator.
        The number of executors in the list corresponds to the number of search objectives.
        Minimum fitness is set to 25.0, which is the minimum path length for the RRT algorithm to identify to count the test as challenging"""

        #self.executor = ObstacleSceneExecutor(self.generator)
        self.executors = [RRTExecutor(self.generator, min_fitness=25.0)]

