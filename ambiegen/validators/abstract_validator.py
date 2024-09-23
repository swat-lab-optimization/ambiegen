import numpy as np
from abc import ABC, abstractmethod
import typing

class AbstractValidator(ABC):
    def __init__(self, config: dict):
        self._config = config


    @abstractmethod
    def is_valid(self) -> typing.Tuple([bool, str]):
        pass

        