import logging
import argparse


log = logging.getLogger(__name__)


def parse_arguments_test_generation():
    """
    This function parses the arguments passed to the script
    :return: The arguments that are being passed to the program
    """

    log.info("Parsing the arguments")
    parser = argparse.ArgumentParser(
        prog="generate_tests.py",
        description="A tool for generating test cases for autonomous systems",
    )
    parser.add_argument(
        "--module-name",
        type=str,
        help="Name of the module containing the test generator class",
        required=True,
    )
    parser.add_argument(
        "--class-name",
        type=str,
        help="Name of the test generator class",
        required=True,
    )

    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to the configuration file",
        required=True,
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        help="Number of runs to perform",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--add-info",
        type=str,
        help="Additional information to add to the result folder name",
        required=False,
        default="",
    )
    in_arguments = parser.parse_args()
    return in_arguments

def parse_arguments_dataset_generation():
    """
    This function parses the arguments passed to the script
    :return: The arguments that are being passed to the program
    """

    log.info("Parsing the arguments")
    parser = argparse.ArgumentParser(
        prog="generate_dataset.py",
        description="A tool for generating test cases for autonomous systems",
    )
    parser.add_argument(
        "--module-name",
        type=str,
        help="Name of the module containing the test generator class",
        required=True,
    )
    parser.add_argument(
        "--class-name",
        type=str,
        help="Name of the test generator class",
        required=True,
    )

    parser.add_argument(
        "--size",
        type=int,
        help="Dataset size",
        required=False,
        default=10000,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Dataset directory",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--tc-dir",
        type=str,
        help="Folder containing the test cases",
        required=False,
        default=None,
    )
    in_arguments = parser.parse_args()
    return in_arguments