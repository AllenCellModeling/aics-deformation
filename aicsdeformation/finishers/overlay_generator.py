import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List
from skvideo.io import FFmpegWriter

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

    def finish(self) -> None:
        """
        This is the function to call from main that will run use the member functions to
        generate a finished deformation movie.
        :return: None
        """
        self.make_heatmap_cell_overlays()
        self.images_to_movie()

    def create_overlay(self, disp: Displacement, fg_path: Path, fg_img: np.ndarray, extent: list) -> Path:
        """
        Create the overlay of displacement and max-projection
        :param disp: The displacement object containing x, y, u, v, mask
        :param fg_path: path to the foreground image (cell max-projection)
        :param fg_img: path to the background image (deformation heatmap)
        :param extent: The x, y - min max bounds (x_min, x_max, y_min, y_max)
        :return: path to the output overlay image
        """
        fig = plt.figure(frameon=False)
        plt.imshow(disp.magnitude_grid, cmap=plt.cm.magma, alpha=.9, interpolation='nearest',
                   extent=extent, vmin=0, vmax=15, origin='lower')
        plt.imshow(fg_img, cmap=plt.cm.gray, alpha=.3, interpolation='bilinear', extent=extent)
        plt.tight_layout(True)
        plt.axis('off')
        fname = self.over_home / fg_path.name
        fig.savefig(fname, bbox_inches='tight')
        fig.clear()
        return fname

    def make_heatmap_cell_overlays(self) -> None:
        """
        This batches through all the time-points to create the deformation / cell overlays
        :return: None
        """
        x_min_v = [np.min(d.x) for d in self.disps]
        x_max_v = [np.max(d.x) for d in self.disps]
        y_min_v = [np.min(d.y) for d in self.disps]
        y_max_v = [np.max(d.y) for d in self.disps]
        extents = zip(x_min_v, x_max_v, y_min_v, y_max_v)

        self.cell_images.set_path_image()  # set the iterator to give path, image
        self.over_images.extend(
            [self.create_overlay(d, p, i, e) for d, (p, i), e in zip(self.disps, self.cell_images, extents)]
        )
        self.cell_images.set_image()  # set the iterator back to the default of returning images

    def images_to_movie(self) -> None:
        """
        Take the overlay images and composite them into a movie
        :return: None
        """
        v_filepath = self.over_home / "movie.mp4"
        writer = FFmpegWriter(v_filepath)
        [writer.writeFrame(img) for img in self.over_images]
        writer.close()
