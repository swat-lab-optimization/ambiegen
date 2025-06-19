"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for saving test scenario images
"""

import os
import logging #as log
from datetime import datetime
from ambiegen.generators.abstract_generator import AbstractGenerator
log = logging.getLogger(__name__)

def save_tcs_images(dt_string, generator: AbstractGenerator, test_suite, config, run, phenotype=False, root_path="experiments"):
    """
    It takes a test suite, a problem, and a run number, and then it saves the images of the test suite
    in the images folder

    Args:
      test_suite: a dictionary of solutions, where the key is the solution number and the value is the
    solution itself
      problem: the problem to be solved. Can be "robot" or "vehicle"
      run: the number of the runs
        algo: the algorithm used to generate the test suite. Can be "random", "ga", "nsga2",
    """

    problem = config["search_based"]["problem_name"]
    algorithm = config["search_based"]["algorithm"]
    name = config["experiment"]["name"]

    images_path = dt_string + "_" + "images" + "_" + algorithm + "_" + problem + "_" + name

    images_path = os.path.join(root_path, images_path)

    if not os.path.exists(images_path):
        os.makedirs(images_path,  exist_ok=True)
    if not os.path.exists(os.path.join(images_path, "run" + str(run))):
        os.makedirs(os.path.join(images_path, "run" + str(run)))

    for i in range(len(test_suite)):

        path = os.path.join(images_path, "run" + str(run))

        test = test_suite[str(i)]
        if phenotype:
            test = generator.genotype2phenotype(test)
            
        test = generator.denormalize_flattened_test(test)
        generator.visualize_test(test, path, i)

    log.info("Images saved in %s", images_path)
