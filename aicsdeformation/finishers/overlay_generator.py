import cv2
import numpy as np
from pathlib import Path
from typing import List

from ..types import Displacement
from ..loaders.path_images import PathImages


class OverlayGenerator:
    """
    This class uses the generated images and displacements (u, v) to create deformation heatmaps
    with cell max-projection overlays
    """
    def __init__(self, displacement_list: List[Displacement], cell_images: PathImages, over_path: Path):
        """
        :param displacement_list: List of displacement objects for each pair of images
        :param cell_images: The PathImages(List) that contains the paths to the max projection images
        :param over_path: The directory Path to save the overlay deformation / cell images to
        """
        self.disps = displacement_list
        self.cell_images = cell_images
        self.over_home = over_path
        self.over_images = PathImages()
        self.min_data_value = 2250  # these are hardcoded for now based on histograms of the deformation data
        self.max_data_value = 4750

    def process(self) -> None:
        """
        This is the function to call from main that will run use the member functions to
        generate a finished deformation movie.
        :return: None
        """
        self.make_heatmap_cell_overlays()
        self.generate_mp4()

    def create_overlay(self, disp: Displacement, fg_path: Path, fg_img: np.ndarray) -> Path:
        """
        Create the overlay of displacement and max-projection
        :param disp: The displacement object containing x, y, u, v, mask
        :param fg_path: path to the foreground image (cell max-projection)
        :param fg_img: path to the background image (deformation heatmap)
        :return: path to the output overlay image
        """
        # hardcode scaling for now
        fgimg = cv2.imread(str(fg_path))
        dmag = np.nan_to_num(disp.magnitude_grid, copy=True)
        dmag = np.uint8(np.clip(255*(dmag-self.min_data_value)/(self.max_data_value - self.min_data_value), 0, 255))
        dx = int((fgimg.shape[0] - dmag.shape[0])/2)
        dy = int((fgimg.shape[1] - dmag.shape[1])/2)
        result = np.zeros(fgimg.shape)
        result[dx:dx+dmag.shape[0], dy:dy+dmag.shape[1], 2] = dmag
        result = np.flip(result, 0)
        result[:, :, 0] = fgimg[:, :, 0]
        fname = self.over_home / fg_path.name
        cv2.imwrite(str(fname), result)
        return fname

    def make_heatmap_cell_overlays(self) -> None:
        """
        This batches through all the time-points to create the deformation / cell overlays
        :return: None
        """
        self.cell_images.set_path_image()  # set the iterator to give path, image
        self.over_images.extend(
            [self.create_overlay(d, p, i) for d, (p, i) in zip(self.disps, self.cell_images)]
        )
        self.cell_images.set_image()  # set the iterator back to the default of returning images

    def generate_mp4(self) -> None:
        """
        Take the overlay images and composite them into a movie
        :return: None
        """
        self.over_images.set_path()
        first = cv2.imread(str(self.over_images[0]))
        height, width, layers = first.shape

        v_filepath = self.over_home / "movie.mpg"
        video = cv2.VideoWriter(filename=str(v_filepath),
                                fourcc=cv2.VideoWriter_fourcc(*'MPEG'),
                                fps=10,
                                frameSize=(width, height))
        for imp in self.over_images:
            print(imp)
            fr = cv2.imread(str(imp))
            video.write(fr)
        video.release()
