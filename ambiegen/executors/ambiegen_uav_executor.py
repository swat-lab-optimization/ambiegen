from ambiegen.executors.abstract_executor import AbstractExecutor
from ambiegen.generators.abstract_generator import AbstractGenerator
import logging
from ambiegen.common.rrt import RRT
#from ambiegen.common.rrt_star import RRTStar
log = logging.getLogger(__name__)

class AmbieGenUAVExecutor(AbstractExecutor):
    '''
    AmbieGenUAVExecutor executes UAV test scenarios using the RRT (Rapidly-exploring Random Tree) algorithm to evaluate test fitness.
    This executor integrates with a generator to define the environment and obstacles, runs the RRT path planning algorithm, and computes a fitness score based on the path length and simulation results. If the RRT fails to find a path or the fitness is below a threshold, it executes the test in simulation, evaluates the minimum distance to obstacles, and applies additional fitness bonuses or penalties. The outcome and metrics are logged for each execution.
    
    Attributes:
        n_sim_evals (int): Counter for the number of simulation evaluations performed.
        _name (str): Name identifier for the executor.
        min_fitness (int): Minimum fitness threshold.
        generator (AbstractGenerator): The generator object providing environment boundaries and obstacle information.
   
    Methods:
        _execute(test) -> float:
            Executes the test scenario using RRT, evaluates the path, and computes the fitness value.
    '''
    def __init__(self, generator: AbstractGenerator):
        super().__init__(generator)
        self.n_sim_evals = 0
        self._name = "RRTExecutor"
        self.min_fitness = 25

    def _execute(self, test) -> float:
        """
        Executes the test scenario using the RRT algorithm and returns the fitness value.
        
        Args:
            test (Test): The test scenario to be executed.
        
        Returns:
            float: The fitness value of the executed test scenario.
        """
        min_x = self.generator.min_position.x
        min_y = self.generator.min_position.y
        max_x = self.generator.max_position.x
        max_y = self.generator.max_position.y

        gx, gy = 0, max_y + 10 # define goal position

        show_animation = False#True
        fitness = 0
        tc_obstacle_list = test.test.simulation.obstacles 
        obstacle_list = []
        bonus = 0
        for obs in tc_obstacle_list:
            obstacle_list.append((obs.position.x, obs.position.y, obs.size.l, obs.size.w, obs.position.r))

        rrt = RRT(
            start=[0, 0],
            goal=[gx, gy],
            rand_area=[min_x+10, max_x-10, min_y - 15, max_y + 15],
            obstacle_list=obstacle_list,
            play_area=[min_x+10, max_x-10, min_y - 15, max_y + 15],
            robot_radius=2
        )
        path = rrt.planning(animation=show_animation)
        if path is not None: 
            fitness = -len(path)
        else:
            fitness = -22

        if fitness < -29:
            try:
                self.n_sim_evals += 1
                log.info(f"Running {self.n_sim_evals} simulation")
                test.execute()

                #if len(test.test_results) > 0:
                distances = test.get_distances()
                distance = min(distances)
                if distance < 1.5:
                    bonus += 10 # addd extra fitness if the test fails
                    num_obs = len(test.test.simulation.obstacles)
                    bonus += 1/num_obs*10 # add extra fitness if the number of obstacles is low

                    self.test_dict[self.exec_counter]["outcome"] = "FAIL"
                else:
                    self.test_dict[self.exec_counter]["outcome"] = "PASS"
                log.info(f"Minimum_distance:{(distance)}")
                self.test_dict[self.exec_counter]["metric"] = distance
                self.test_dict[self.exec_counter]["info"] = "simulation"
                test.plot()
            except Exception as e:
                self.test_dict[self.exec_counter]["info"] = "ERROR"
                log.info("Exception during test execution")
                log.info(f"{e}")

        fitness -= bonus
        
        return fitness
            
