import logging #as log
from ambiegen.executors.abstract_executor import AbstractExecutor
from ambiegen.common.vehicle_evaluate import evaluate_scenario
log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130

class SimpleVehicleExecutor(AbstractExecutor):
    """
    Executor class for evaluating vehicle-based test scenarios.
    This class extends AbstractExecutor to provide a simple interface for executing
    vehicle-related tests and retrieving their fitness scores.
    
    Attributes:
        _name (str): Name identifier for the executor.
    
    Methods:
        __init__(generator, results_path=None):
            Initializes the executor with a scenario generator and optional results path.
        _execute(test) -> float:
            Executes the provided test scenario, evaluates its fitness, logs the result,
            and returns the fitness score.
    """

    def __init__(self, generator, results_path: str = None):
        super().__init__(generator, results_path)
        self._name = "SimpleVehicleExecutor"

    def _execute(self, test) -> float:

        fitness, _ = evaluate_scenario(test)

        log.info(f"Fitness: {fitness}")
        #fitness = 0 
        
        return fitness



