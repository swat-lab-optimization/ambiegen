import abc
import typing
import numpy as np
import typing 
import random
from shapely.geometry import Polygon
from ambiegen.generators.abstract_generator import AbstractGenerator
import os
import logging #as log
from aerialist.px4.obstacle import Obstacle
from aerialist.px4.drone_test import DroneTest
from ambiegen.common.testcase import TestCase
from shapely import geometry
from numpy import dot
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

log = logging.getLogger(__name__)

class ObstacleGenerator(AbstractGenerator):
    '''
    ObstacleGenerator is a generator class for creating and manipulating obstacle configurations for drone test cases.
    This class provides methods to randomly generate, normalize, denormalize, and validate obstacle test cases within specified bounds. It supports conversion between phenotype (test case with obstacles) and genotype (flattened, normalized representation), as well as visualization of generated test cases.
    
    Attributes:
        min_size (Obstacle.Size): Minimum allowed size for obstacles.
        max_size (Obstacle.Size): Maximum allowed size for obstacles.
        min_position (Obstacle.Position): Minimum allowed position for obstacles.
        max_position (Obstacle.Position): Maximum allowed position for obstacles.
        case_study (DroneTest): The drone test case loaded from a YAML file.
        max_box_num (int): Maximum number of obstacles allowed in a test case.
        _l_b (np.ndarray): Lower bounds for the flattened test case.
        _u_b (np.ndarray): Upper bounds for the flattened test case.
        _size (int): Size of the genotype vector.
        size (int): Returns the size of the genotype vector.
        lower_bound (list): Returns the lower bound for the genotype vector.
        upper_bound (list): Returns the upper bound for the genotype vector.
    
    Methods:
        cmp_func(x, y): Computes the cosine similarity difference between two vectors.
        get_bounds(): Returns the lower and upper bounds for the genotype vector.
        flatten_test_case(test): Flattens a nested test case structure into a 1D numpy array.
        generate_random_test(): Generates a random, valid obstacle test case and returns its genotype.
        normalize_flattened_test(test): Normalizes a flattened test case using min and max bounds.
        denormalize_flattened_test(norm_test): Denormalizes a normalized test case back to original scale.
        phenotype2genotype(phenotype): Converts a phenotype (TestCase) to a normalized genotype vector.
        genotype2phenotype(genotype): Converts a genotype vector back to a phenotype (TestCase).
        resize_test(test): Reshapes a flattened test case into the expected structure.
        is_valid(test): Checks if a given test case is valid (no overlaps, within bounds).
        obstacles_fit(box_list): Checks if a list of obstacles fit within the defined area and do not overlap.
        get_random_box_vals(): Generates random values for a single obstacle's parameters.
        visualize_test(test, save_path, num, title): Visualizes a test case and saves the plot to a file.
    '''
    def __init__(self,  case_study_file: str, max_box_num:int=3):
        """Initialize the generator.

        Args:
            config (dict): Dictionary containing the configuration parameters.
        """
        super().__init__()
        self.min_size = Obstacle.Size(2, 2, 15)
        self.max_size = Obstacle.Size(20, 20, 25)
        self.min_position = Obstacle.Position(-40, 10, 0, 0)
        self.max_position = Obstacle.Position(30, 40, 0, 90)
        self.case_study = DroneTest.from_yaml(case_study_file)
        self.max_box_num = max_box_num
        self._l_b, self._u_b = self.get_bounds()
        self._size = self.max_box_num*6 + 1 

    @property
    def size(self):
        return self._size
    

    @property
    def lower_bound(self):
        return self._size*[0]
    
    @property
    def upper_bound(self):  
        return self._size*[1]

    def cmp_func(self, x, y):

        self.novelty_name = "cosine"

        cos_sim = dot(x, y) / (norm(x) * norm(y))
        difference = 1 - cos_sim
        return (difference)
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        l_b = [1]
        u_b = [self.max_box_num]
        l_b_ = [self.min_size.l, self.min_size.w, self.min_size.h, self.min_position.x, self.min_position.y,  self.min_position.r]
        u_b_ = [self.max_size.l, self.max_size.w, self.max_size.h, self.max_position.x, self.max_position.y, self.max_position.r]

        for i in range(self.max_box_num):
            l_b.append(l_b_)
            u_b.append(u_b_)


        l_b = self.flatten_test_case(l_b)
        u_b = self.flatten_test_case(u_b)

        return l_b, u_b

    
    def flatten_test_case(self, test) -> np.ndarray:
        result = []
        for item in test:
            if isinstance(item, list):
                result.extend(self.flatten_test_case(item))
            else:
                result.append(item)
        return np.array(result)
    
    def generate_random_test(self) -> typing.List[float]:

        obstacles_list = []
        num_boxes = np.random.choice(np.arange(1, self.max_box_num+1))

        while len(obstacles_list) < (self.max_box_num):
            size = Obstacle.Size(
            l=random.choice(np.arange(self.min_size.l, self.max_size.l)),
            w=random.choice(np.arange(self.min_size.w, self.max_size.w)),
            h=random.choice(np.arange(self.min_size.h, self.max_size.h)),
            )
            position = Obstacle.Position(
            x=random.choice(np.arange(self.min_position.x, self.max_position.x)),
            y=random.choice(np.arange(self.min_position.y, self.max_position.y)),
            z=0,  # obstacles should always be place on the ground
            r=random.choice(np.arange(self.min_position.r, self.max_position.r)),
            )
            obstacle = Obstacle(size, position)

            to_include = self.obstacles_fit([obstacle] + obstacles_list)
            if to_include:
                obstacles_list.append(obstacle)


        obstacles_list = obstacles_list[:num_boxes]
        the_test = TestCase(self.case_study, obstacles_list)
        self.phenotype = the_test
        self.genotype = self.phenotype2genotype(the_test)

        return self.genotype
        
    def normalize_flattened_test(self, test: typing.List[float]) -> typing.List[float]:
        '''
        Use min and max values to normalize the test case.
        Args:
            test (list): Flattened test case to normalize.'''
        result = (test - self._l_b)/(self._u_b - self._l_b)
        return result
    
    def denormalize_flattened_test(self, norm_test: typing.List[float]) -> typing.List[float]:
        '''
        Use min and max values to denormalize the test case.
        Args:
            norm_test (list): Flattened normalized test case to denormalize.
        '''
        result = norm_test*(self._u_b - self._l_b) + self._l_b
        result =  np.array(result, dtype=object)
        result[0] = int(round(result[0]))  # Ensure the first element is an integer (number of boxes)

        return result
    
    def phenotype2genotype(self, phenotype: TestCase) -> typing.List[float]:
        obstacles_list = phenotype.test.simulation.obstacles
        num_boxes = len(obstacles_list)
        tc = [num_boxes]
        for b in obstacles_list:
            tc.extend([b.size.l, b.size.w, b.size.h, b.position.x, b.position.y, b.position.r])

        for r in range(num_boxes, self.max_box_num):
            tc.extend(self.get_random_box_vals())   

        tc = self.normalize_flattened_test(tc)

        return tc
        
    def genotype2phenotype(self, genotype: typing.List[float]) -> TestCase:

        denormilized_tc = self.denormalize_flattened_test(genotype)
        resized_tc = self.resize_test(denormilized_tc)
        num_boxes = min(resized_tc[0], self.max_box_num)
        tc = resized_tc[1]
        obstacles_list = []
        for b in range(num_boxes):
            size = Obstacle.Size(
            l=tc[b][0],
            w=tc[b][1],
            h=tc[b][2],
            )
            position = Obstacle.Position(
            x=tc[b][3],
            y=tc[b][4],
            z=0,  # obstacles should always be place on the ground
            r=tc[b][5],
            )
            obstacle = Obstacle(size, position)

            obstacles_list.append(obstacle)

        the_test = TestCase(self.case_study, obstacles_list)

        return the_test
    
    def resize_test(self, test):
        num_boxes = int(round(test[0]))
        test = test[1:]
        test = test.reshape(-1, 6)
        return [num_boxes, test]
    

    def is_valid(self, test) -> tuple[bool, str]:
        """
        Check if the given test is valid.

        Args:
            test: The test to be validated.

        Returns:
            A tuple containing a boolean indicating the validity of the test and a string with additional information.
        """
        #log.info(f"test {test}")
        boxes  = test.test.simulation.obstacles
        
        valid, info  = self.obstacles_fit(boxes)

        return valid, info


    def obstacles_fit(self, box_list: typing.List[Obstacle]) -> tuple[bool, str]:
        """
        Check if the boxes in the given list fit within the defined bounds.

        Args:
            box_list: A list of boxes to be checked.

        Returns:
            A tuple containing a boolean indicating if the boxes fit within the bounds and a string with additional information.
        """
        existing_boxes_geometry_list = [obstacle.geometry for obstacle in box_list]#[obstacle.position.x, obstacle.position.y, obstacle.position.r]

        min_pos = [self.min_position.x, self.min_position.y]
        max_pos = [self.max_position.x, self.max_position.y]

        outer_polygon = geometry.Polygon([min_pos, [min_pos[0], max_pos[1]], max_pos, [max_pos[0], min_pos[1]]])

        n = len(existing_boxes_geometry_list)

        for i in range(n):
            for j in range(i + 1, n):
                poly1 = existing_boxes_geometry_list[i]
                poly2 = existing_boxes_geometry_list[j]

                # Check if polygons intersect
                if poly1.intersects(poly2):
                    
                    return False, f"Obstacles {i} and {j} intersect."

        for inner_polygon in existing_boxes_geometry_list:
            if not(inner_polygon.within(outer_polygon)):
                return False, "Obstacle out of the defined bounds"
            

        return True, "Valid test"
    
    def get_random_box_vals(self):
        l = random.choice(np.arange(self.min_size.l, self.max_size.l))
        w = random.choice(np.arange(self.min_size.w, self.max_size.w))
        h = random.choice(np.arange(self.min_size.h, self.max_size.h))
        x = random.choice(np.arange(self.min_position.x, self.max_position.x))
        y = random.choice(np.arange(self.min_position.y, self.max_position.y))
        #z=0  # obstacles should always be place on the ground
        r = random.choice(np.arange(self.min_position.r, self.max_position.r))
        return [l, w, h, x, y, r]

    def visualize_test(self, test,  save_path:str = "test", num=0, title="", drone_path=None):
            #test.plot()
        obstacle_num = round(test[0])
        obstacles = np.array(test[1:6*obstacle_num+1])
        obstacles = obstacles.reshape(-1, 6)

        start_point = [0, 0]
        end_point = [0, 50]
    
    # Convert the numpy array back to a list of lists
        obstacles = obstacles.tolist()
        #obstacles = test.test.simulation.obstacles
        fig, ax = plt.subplots(figsize=(8,5.7))

        ax.set_xlim(self.min_position[0], self.max_position[0]+10) #80
        ax.set_ylim(self.min_position[1] -12, self.max_position[1] + 15) # 57

        area_x = (self.min_position[0] + self.max_position[1])/2
        area_y = (self.min_position[1] + self.max_position[1])/2

        area_width = self.max_position[0] - self.min_position[0]
        area_height = self.max_position[1] - self.min_position[1]

        rect = patches.Rectangle((area_x  -area_width/2, area_y - area_height/2), area_width, area_height, linewidth=1, edgecolor='black', facecolor='none', label='Obstacle area')
        ax.add_patch(rect)

        ax.scatter(start_point[0], start_point[1], c='green', label='Start point')
        ax.scatter(end_point[0], end_point[1], c='blue', label='End point')


        if obstacles is not None:
            for obst in obstacles:
                length, width, height, x, y, r = obst
                rect = patches.Rectangle(
                    (x - length / 2, y - width/ 2),  # bottom-left corner
                    length,
                    width,
                    angle=r,
                    edgecolor='blue',
                    facecolor='none',
                    rotation_point="center"
                )

                # Add the rectangle patch to the Axes
                ax.add_patch(rect)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16)
        plt.ioff()
        ax.set_title(title, fontsize=16)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)

        if drone_path is not None:
            ax.scatter(drone_path[0], drone_path[1], c='red', label='Drone path')

        #ax.legend()

        if not(os.path.exists(save_path)):
            os.makedirs(save_path, exist_ok=True)
        #print("Save path", save_path)
        final_path = os.path.join(save_path, str(num) + ".png")
        fig.savefig(final_path, bbox_inches='tight')
        log.info("Saved image to " + final_path)
        print("Saved image to " + final_path)
        plt.close(fig)

        

        

        



        

        
        

