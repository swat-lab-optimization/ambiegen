from ambiegen.executors.abstract_executor import AbstractExecutor
import logging
from ambiegen.common.rrt import RRT
log = logging.getLogger(__name__)

class RRTExecutor(AbstractExecutor):
    '''
    RRTExecutor executes test scenarios using the Rapidly-exploring Random Tree (RRT) algorithm to evaluate their fitness.
    This executor interfaces with a generator to obtain environment boundaries and obstacles, then attempts to find a path from a fixed start position to a goal using RRT. The fitness of a test scenario is determined by the length of the path found (shorter paths yield higher fitness), or zero if no path is found.
    
    Attributes:
        n_sim_evals (int): Counter for the number of simulation evaluations performed.
        _name (str): Name identifier for the executor.
        min_fitness (float): Minimum fitness threshold for test scenarios.
        generator: An object providing environment boundaries and obstacle information.
        min_fitness (float, optional): Minimum fitness threshold. Defaults to 25.0.
    
    Methods:
        _execute(test) -> float:
            Executes the RRT algorithm on the provided test scenario and returns its fitness value.
    '''

    def __init__(self, generator, min_fitness: float = 25.0):
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
            fitness = 0

        return fitness
            
