from abc import ABC, abstractmethod
import numpy as np


class LoaderABC(ABC):

    @abstractmethod
    def generate_bead_imgs(self, ti: int, dcube: np.ndarray):
        pass

    @abstractmethod
    def generate_projection_imgs(self, ti: int, dcube: np.ndarray):
        pass
