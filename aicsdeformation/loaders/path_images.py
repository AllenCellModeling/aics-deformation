from enum import Enum, unique
from imageio import imread


class PathImages(list):
    """BeadImages is a subclass of list. The intent is to store paths and then just load the image when the list
    element is requested rather than keeping all the list elements in memory simultaneously. """
    @unique
    class _RType(Enum):
        """
        Enum defining valid return state_types of PathImages
        """
        PATH = 1
        IMAGE = 2
        PATH_IMAGE = 3

    class RTypeException(TypeError):
        """
        Custom exception in case someone sets the return state_type to an ill defined value.
        """
        pass

    def __init__(self, *args):
        super().__init__(*args)
        self.return_type = PathImages._RType.IMAGE

    def set_image(self) -> None:
        """
        Set the iterator to evaluate an image not the path
        :return: None
        """
        self.return_type = PathImages._RType.IMAGE

    def set_path(self) -> None:
        """
        Set the iterator to evaluate a path
        :return: None
        """
        self.return_type = PathImages._RType.PATH

    def set_path_image(self) -> None:
        """
        Set the iterator to evaluate (path, image)
        :return: None
        """
        self.return_type = PathImages._RType.PATH_IMAGE

    def set_default(self):
        self.set_image()

    def __getitem__(self, index: int):
        return self.my_return(super().__getitem__(index))

    def __iter__(self):
        for pth in super().__iter__():
            yield self.my_return(pth)

    def my_return(self, pth):
        """
        Read the object's state (self.return_type) and apply imread when appropriate
        :param pth: The path super's iterator returns
        :return: a path, image or both depending on self.return_type
        """
        if self.return_type is PathImages._RType.IMAGE:
            return imread(pth)
        if self.return_type is PathImages._RType.PATH:
            return pth
        if self.return_type is PathImages._RType.PATH_IMAGE:
            return pth, imread(pth)
        raise PathImages.RTypeException()
