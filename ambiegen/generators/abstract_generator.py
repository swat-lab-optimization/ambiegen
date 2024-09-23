import abc
import typing
import numpy as np

class AbstractGenerator(abc.ABC):
    """Abstract class for all generators."""
    
    def __init__(self, config: dict, u_b:typing.List[float], l_b:typing.List[float], n_dim:int):
        """Initialize the generator with a configuration dictionary.

        Args:
            config (dict): Dictionary containing the configuration parameters.
        """
        self._name = "AbstractGenerator"
        self._config = config
        self._phenotype = None
        self._genotype_size = n_dim
        self._u_b = u_b
        self._l_b = l_b
        assert len(u_b) == len(l_b) == n_dim, "Upper and lower bounds should have the same length as the genotype size."
        self._genotype = None

    @property
    def genotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self._genotype_size

    @property
    def name(self) -> str:
        """Name of the generator.

        Returns:
            str: Name of the generator.
        """
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set the name of the generator.

        Args:
            value (str): Name of the generator.
        """
        self._name = value

    @abc.abstractmethod
    def cmp_func(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compare two genotypes.

        Args:
            x (np.ndarray): First genotype.
            y (np.ndarray): Second genotype.

        Returns:
            float: Comparison result.
        """
        pass

    @abc.abstractmethod
    def genotype2phenotype(self, genotype: typing.List[float]):
        """Convert a genotype to a phenotype. This a method to convert the actual
        test representation i.e. test .yaml file to a format that can be used by the
        genetic algorithm.

        Args:
            genotype (typing.List[float]): The genotype to convert.

        Returns:
            The resulting phenotype.
        """
        pass

    @abc.abstractmethod
    def generate_random_test(self) -> typing.Tuple[typing.List[float], typing.List[int]]:
        """Generate a random valid test. This method should generate a random vaid test and return it.
        It should also set 

        Returns:
            typing.Tuple[typing.List[float], typing.List[int]]: The generated test and a success flag.
        """
        pass

    @abc.abstractmethod
    def visualize_test(self, test: typing.List[float], save_path: str = None):
        """Visualize a test.

        Args:
            test (typing.List[float]): Test to visualize.
            save_path (str, optional): Path to save the visualization. Defaults to None.
        """
        pass
