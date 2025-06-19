import abc
import typing
import numpy as np


class AbstractGenerator(abc.ABC):
    '''
    Abstract base class for genotype-phenotype generators.
    This class defines the interface for generators that map between genotype and phenotype
    representations, provide bounds for genotypes, generate random samples, validate tests,
    and visualize them. Subclasses must implement all abstract properties and methods.
    In other words, when adding a new generator, you should implement all the abstract methods defined in this class
    
    Attributes:
        name (str): Name identifier for the generator.
        size (int): Number of elements in the genotype representation.
        lower_bound (List[float]): Lower bounds for each element in the genotype.
        upper_bound (List[float]): Upper bounds for each element in the genotype.
    
    Methods:
        cmp_func(test1: List[float], test2: List[float]) -> float:
            Compare two tests and return -1, 0, or 1.
        genotype2phenotype(genotype: List[float]) -> List[float]:
            Convert a genotype to its phenotype representation.
        generate_random_test() -> List[float]:
            Generate a random valid test (genotype).
        is_valid(test: List[float]) -> bool:
            Check if a test is valid.
        visualize_test(test: List[float], save_path: str = None):
            Visualize a test.
        phenotype2genotype(phenotype: List[float]) -> List[float]:
            Convert a phenotype to its genotype representation.
    '''

    def __init__(self, name: str = "AbstractGenerator"):
        """
        Initialize the generator.

        Args:
            name: Name identifier for the generator
            config: Optional configuration dictionary containing parameters
        """
        self._name = name

    @property
    def name(self) -> str:
        """
        Get the name of the generator.

        Returns:
            Name of the generator
        """
        return self._name

    # Abstract properties that must be implemented by subclasses
    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
        Size of the genotype representation.

        The genotype is the internal representation used by the optimization algorithm.
        This is typically a list/array of floating-point values.

        Returns:
            Number of elements in the genotype
        """
        pass

    @property
    @abc.abstractmethod
    def lower_bound(self) -> typing.List[float]:
        """
        Lower bound of the genotype.

        This is the minimum value for each element in the genotype.

        Returns:
            List of lower bounds for each element in the genotype
        """
        pass

    
    @property
    @abc.abstractmethod
    def upper_bound(self) -> typing.List[float]:
        """
        Upper bound of the genotype.

        This is the maximum value for each element in the genotype.

        Returns:
            List of upper bounds for each element in the genotype
        """
        pass

    @abc.abstractmethod
    def cmp_func(self, test1: typing.List[float], test2: typing.List[float]) -> int:
        """Compare two tests.

        Args:
            test1 (np.array): First test to compare.
            test2 (np.array): Second test to compare.

        Returns:
            float: the degree of similarity between the two tests based on the distance function (cosine similarity by default)
        """
        pass

    def phenotype2genotype(self, phenotype: typing.List[float]) -> typing.List[float]:
        """Convert a phenotype to a genotype.

        Args:
            phenotype (np.array): Phenotype to convert.

        Returns:
            np.array: Genotype representation of the phenotype.
        """
        pass

    @abc.abstractmethod
    def genotype2phenotype(self, genotype: typing.List[float]) -> typing.List[float]:
        """Convert a genotype to a phenotype.

        Args:
            genotype (np.array): Genotype to convert.

        Returns:
            np.array: Phenotype representation of the genotype.
        """
        pass

    @abc.abstractmethod
    def generate_random_test(self) -> (typing.List[float]):
        """Generate samples from the generator

        Returns:
            np.array: Generated samples.
        """
        pass
    @abc.abstractmethod
    def is_valid(self, test: typing.List[float]) -> bool:
        """Check if a test is valid.

        Args:
            test (np.array): Test to check.

        Returns:
            bool: True if the test is valid, False otherwise.
        """
        pass

    @abc.abstractmethod
    def visualize_test(self, test: typing.List[float], save_path: str = None):
        """Visualize a test.

        Args:
            test (np.array): Test to visualize.
        """
        pass

    def __str__(self) -> str:
        """String representation of the generator."""
        return f"{self.__class__.__name__}(name='{self.name}', genotype_size={self.genotype_size})"
