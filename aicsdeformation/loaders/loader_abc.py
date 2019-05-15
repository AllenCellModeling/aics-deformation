from abc import ABC, abstractmethod
from typing import List


class LoaderABC(ABC):

    @abstractmethod
    def process(self) -> List:
        pass
