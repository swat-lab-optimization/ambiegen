from ambiegen.executors.abstract_executor import AbstractExecutor
from ambiegen.generators.abstract_generator import AbstractGenerator
import logging
log = logging.getLogger(__name__)
class ObstacleSceneExecutor(AbstractExecutor):
    """
    Executes obstacle scene tests using a provided generator and evaluates their fitness.
    This executor runs UAV (Unmanned Aerial Vehicle) simulation tests, tracks their outcomes, and computes a fitness score based on the minimum distance to obstacles encountered during the test. It maintains dictionaries to store test results and statistics, such as the number of simulation evaluations and failures.
    
    Arguments:
        generator (AbstractGenerator): The generator used to create test scenarios.
        min_fitness (float, optional): The minimum fitness threshold. Defaults to 0.0.
    
    Attributes:
        uav_test_dict (dict): Stores detailed information about each UAV test execution.
        n_sim_evals (int): Counter for the number of simulation evaluations performed.
        num_failures (int): Counter for the number of failed tests.
    
    Methods:
        _execute(test) -> float:
            Executes a single test, records its results, and returns a fitness value.
            The fitness is calculated as the negative inverse of the minimum distance to obstacles.
            Handles exceptions during test execution and logs errors.
    """

    def __init__(self, generator: AbstractGenerator, min_fitness: float = 0.0):
        super().__init__(generator, min_fitness)
        self.uav_test_dict = {}
        self.n_sim_evals = 0
        self.num_failures = 0


    def _execute(self, test) -> float:
        fitness = 0
        self.uav_test_dict[self.exec_counter] = {}
        self.uav_test_dict[self.exec_counter]["test"] = test

        try:
            self.n_sim_evals += 1
            trajectory = test.execute()
            data = trajectory.to_data_frame()
            #print(data)
            x_coord = list(data[:, 1])
            y_coord = list(data[:, 2])
            z_coord = list(data[:, 3])
            yaw = list(data[:, 4])


            if len(test.test_results) > 0:

                distances = test.get_distances()
                distance = min(distances)
                if distance < 1.5:
                    self.test_dict[self.exec_counter]["outcome"] = "FAIL"
                    self.num_failures += 1
                    self.test_dict[self.exec_counter]["features"] = {}
                    self.test_dict[self.exec_counter]["features"]["x_coord"] = x_coord
                    self.test_dict[self.exec_counter]["features"]["y_coord"] = y_coord
                    self.test_dict[self.exec_counter]["features"]["z_coord"] = z_coord
                    self.test_dict[self.exec_counter]["features"]["yaw"] = yaw
                else:
                    self.test_dict[self.exec_counter]["outcome"] = "PASS"
                log.info(f"Minimum_distance:{(distance)}")
                self.test_dict[self.exec_counter]["metric"] = distance
                self.test_dict[self.exec_counter]["num_failures"] = self.num_failures
                self.test_dict[self.exec_counter]["sim_num"] = self.n_sim_evals


                self.uav_test_dict[self.exec_counter]["info"] = "simulation"

                fitness = -1/distance
                test.plot()
        except Exception as e:
                self.test_dict[self.exec_counter]["info"] = "ERROR"
                self.uav_test_dict[self.exec_counter]["info"] = "ERROR"
                log.info("Exception during test execution, skipping the test")
                log.info(f"{e}")

        return fitness
            