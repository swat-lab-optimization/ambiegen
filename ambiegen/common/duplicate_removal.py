from pymoo.core.duplicate import ElementwiseDuplicateElimination
import numpy as np

import math
import logging 
from numpy import dot
from numpy.linalg import norm
from ambiegen.generators.abstract_generator import AbstractGenerator
log = logging.getLogger(__name__)
import time


class AbstractDuplicateElimination(ElementwiseDuplicateElimination):


    def __init__(self, generator:AbstractGenerator, threshold:float = 0.15):
        super().__init__()
        self.generator = generator
        self.threshold = threshold

    def is_equal(self, a, b):
        vector1 = np.array(a.X)
        vector2 = np.array(b.X)
        difference = self.generator.cmp_func(vector1, vector2)
        return difference < self.threshold

