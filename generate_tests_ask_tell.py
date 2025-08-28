import importlib
import logging
import os
import shutil
from datetime import datetime
from typing import Type, Optional
import yaml
import traceback
from ambiegen.common.save_tc_results import save_tc_results, create_summary
from ambiegen.common.save_tcs_images import save_tcs_images
from ambiegen.common.parse_arguments import parse_arguments_test_generation
from ambiegen.common.get_convergence import get_convergence
from ambiegen.common.get_stats import get_stats
from ambiegen.common.get_test_suite import get_test_suite


logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").disabled = True
log = logging.getLogger(__name__)

def setup_logging(debug: bool = False, log_to: Optional[str] = "log.txt") -> None:
    """
    Set up the logging system.
    """
    term_handler = logging.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to:
        file_handler = logging.FileHandler(log_to, "w", "utf-8")
        log_handlers.append(file_handler)
        start_msg += f", writing logs to file: {log_to}"

    log_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )

    logging.info(start_msg)


def get_fitness_from_simulator():
    return 1

def generate_tests(
    runs: int = 1,
    generator_class: Type = None,
    config_file = None,
) -> None:
    """
    Run the optimization process.

    Usage:
        python generate_tests.py --module-name "ambiegen.testers.uav_tester" --class-name "UAVTester" --runs 3 --config-path "tester_config.yaml"
    """
    log.info("Starting optimization")
    log.info(f"Number of runs: {runs}")
    log.info(f"Generator: {generator_class}")
    log.info(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    log.info("----------------------------------------------------")

    tc_stats, tcs, tcs_convergence, all_tests = {}, {}, {}, {}
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M")
    root_path = os.path.join("experiments", )
    log.info(f"Saving results to: {root_path}")

    for run in range(runs):
        log.info(f"Run {run}")

        tester = generator_class(config_file)
        tester.initialize()

        for num in range(15):

            try:
                x = tester.ask_next()
                valid, fitness, test = tester.verify_next(x)
                if valid:
                    fitness = [get_fitness_from_simulator()]
                tester.tell_next(x, fitness, "pass")
            except Exception as e:
                log.error(f"Error while running generator: {e}")
                log.info(f"Error {traceback.format_exc()} found")
                log.error("Error during execution of test.", exc_info=True)
                exit(1)

        log.info(f"Run {run} finished")
        log.info("----------------------------------------------------")


        generated_tests, res = tester.get_results()

        # tc_stats[f"run{run}"] = get_stats(res)
        # tcs_convergence[f"run{run}"] = get_convergence(res)
        # tcs[f"run{run}"] = get_test_suite(res)
        all_tests[f"run{run}"] = generated_tests

        # save_tc_results(
        #     dt_string,
        #     tc_stats,
        #     tcs,
        #     tcs_convergence,
        #     all_tests,
        #     config,
        #     root_path=root_path,
        # )
        #save_tcs_images(dt_string, tester.generator, tcs[f"run{run}"],config, run, root_path=root_path)

if __name__ == "__main__":
    args = parse_arguments_test_generation()
    module_name = args.module_name
    class_name = args.class_name
    runs = args.runs
    config_path = args.config_path


    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    setup_logging(debug=False, log_to="log.txt")
    module = importlib.import_module(module_name)
    the_class = getattr(module, class_name)

    generate_tests(runs=runs, generator_class=the_class, config_file=config)
