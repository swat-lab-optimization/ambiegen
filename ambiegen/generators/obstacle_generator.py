
from aerialist.px4.obstacle import Obstacle
from aerialist.px4.drone_test import DroneTest
from ambiegenvae.common.testcase import TestCase
from shapely import geometry
from numpy import dot
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt 

import abc
import typing
import numpy as np
import typing 
import random
from shapely.geometry import Polygon
from ambiegenvae.generators.abstract_generator import AbstractGenerator
import yaml
#import cv2
import os
import logging #as log
log = logging.getLogger(__name__)

class ObstacleGenerator(AbstractGenerator):
    """Abstract class for all generators."""
    def __init__(self, min_size:Obstacle, max_size:Obstacle, min_position:Obstacle, max_position:Obstacle, case_study_file: str, max_box_num:int=3):
        """Initialize the generator.

        Args:
            config (dict): Dictionary containing the configuration parameters.
        """
        super().__init__()
        self.min_size = min_size #[min_size.l, min_size.w, min_size.h]
        self.max_size = max_size #[max_size.l, max_size.w, max_size.h]
        self.min_position = min_position #[min_position.x, min_position.y, 0, min_position.r]
        self.max_position = max_position #[max_position.x, max_position.y, 0, max_position.r]
        #print(case_study_file)
        #print(os.getcwd())
        self.case_study = DroneTest.from_yaml(case_study_file)
        self.max_box_num = max_box_num
        self._size = self.max_box_num*6 + 1
        self._size = self.max_box_num*6 + 1
        self.l_b, self.u_b = self.get_bounds()
        self.l_b_norm = np.array(self._size * [0])
        self.u_b_norm = np.array(self._size * [1])
        self._genotype = None    

    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, size):
        self._size = size

    
    
    @property
    def phenotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self.size#max_number_of_points
    

    def cmp_func(self, x, y):
        cos_sim = dot(x, y) / (norm(x) * norm(y))

        difference = 1 - abs(cos_sim)
        return difference
        

    def get_bounds(self):
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
    
    def flatten_test_case(self, test):
        result = []
        for item in test:
            if isinstance(item, list):
                result.extend(self.flatten_test_case(item))
            else:
                result.append(item)
        return np.array(result)
    
    def get_random_box_vals(self):
        l=random.choice(np.arange(self.min_size.l, self.max_size.l))
        w=random.choice(np.arange(self.min_size.w, self.max_size.w))
        h=random.choice(np.arange(self.min_size.h, self.max_size.h))
        x=random.choice(np.arange(self.min_position.x, self.max_position.x))
        y=random.choice(np.arange(self.min_position.y, self.max_position.y))
        #z=0  # obstacles should always be place on the ground
        r=random.choice(np.arange(self.min_position.r, self.max_position.r))
        return [l, w, h, x, y, r]

    def generate_random_test(self, genotype=True):

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

            to_include = self.obstacle_fits(obstacle, obstacles_list)
            if to_include:
                obstacles_list.append(obstacle)


        obstacles_list = obstacles_list[:num_boxes]

        #print("Genotype", self.genotype)
        the_test = TestCase(self.case_study, obstacles_list)

       # self.set_genotype(self.phenotype2genotype(the_test))
        self.genotype = self.phenotype2genotype(the_test)

        return the_test, True
    
    def normilize_flattened_test(self, test):
        result = (test - self.l_b)/(self.u_b - self.l_b)
        return result
    
    def denormilize_flattened_test(self, norm_test):
        result = norm_test*(self.u_b - self.l_b) + self.l_b
        return result
    
    def phenotype2genotype(self, phenotype):
        obstacles_list = phenotype.test.simulation.obstacles
        num_boxes = len(obstacles_list)
        tc = [num_boxes]
        for b in obstacles_list:
            tc.extend([b.size.l, b.size.w, b.size.h, b.position.x, b.position.y, b.position.r])

        #for r in range(num_boxes, self.max_box_num): # extent with empty values
        #    tc.extend([self.min_size.l, self.min_size.w, self.min_size.h, self.min_position.x, self.min_position.y, self.min_position.r])
        for r in range(num_boxes, self.max_box_num):
            tc.extend(self.get_random_box_vals())   

        tc = self.normilize_flattened_test(tc)

        return tc
    
    @property
    def genotype(self):
        return self._genotype
    

    @genotype.setter
    def genotype(self, genotype):
       self._genotype = genotype

    def get_genotype(self):
        return self._genotype
    
    def get_phenotype(self):
        self.phenotype = self.genotype2phenotype(self.genotype)
        return self.phenotype
    

    def resize_test(self, test):
        num_boxes = int(round(test[0]))
        #print(f"num_boxes {num_boxes}")
        test = test[1:]
        # resize test to the shape (max_box_num, 7)
        test = test.reshape(-1, 6)

        return [num_boxes, test]
        
    def genotype2phenotype(self, genotype):

        denormilized_tc = self.denormilize_flattened_test(genotype)
        #print("Denormilized tc", denormilized_tc)
        resized_tc = self.resize_test(denormilized_tc)
        num_boxes = min(resized_tc[0], self.max_box_num)
        tc = resized_tc[1]
        obstacles_list = []
        #print("Boxes num", num_boxes)
        #print("tc", tc)
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


    def obstacle_fits(self, obstacle:Obstacle, obstacles_list:list):

        new_box_geometry = obstacle.geometry#[obstacle.size.l, obstacle.size.w, obstacle.size.h]
        existing_boxes_geometry_list = [obstacle.geometry for obstacle in obstacles_list]#[obstacle.position.x, obstacle.position.y, obstacle.position.r]

        min_pos = [self.min_position.x, self.min_position.y]
        max_pos = [self.max_position.x, self.max_position.y]

        outer_polygon = geometry.Polygon([min_pos, [min_pos[0], max_pos[1]], max_pos, [max_pos[0], min_pos[1]]])
        

        for box in existing_boxes_geometry_list:
            if new_box_geometry.intersects(box):
                return False
        is_inside = new_box_geometry.within(outer_polygon)
        if not(is_inside):
            return False
        return True
    
    def visualize_test(self, test,  save_path:str = "test.png", num=0, title=""):
        #test.plot()
        obstacles = test.test.simulation.obstacles
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_xlim(self.min_position[0], self.max_position[1])
        ax.set_ylim(self.min_position[1], self.max_position[1])


        if obstacles is not None:
            for obst in obstacles:
                obst_patch = obst.plt_patch()
                obst_patch.set_label("obstacle")
                ax.add_patch(obst_patch)


        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16)
        plt.ioff()
        ax.set_title(title, fontsize=16)

        #ax.legend()

        if not(os.path.exists(save_path)):
            os.makedirs(save_path, exist_ok=True)
        final_path = os.path.join(save_path, str(num) + ".png")
        fig.savefig(final_path, bbox_inches='tight')
        log.info("Saved image to " + final_path)
        print("Saved image to " + final_path)
        plt.close(fig)


    
        

        

        



        

        
        

