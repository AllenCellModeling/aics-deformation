from enum import IntEnum
import numpy as np
from pathlib import Path
from typing import List, Tuple

from aicsimageio.omeTifWriter import OmeTifWriter

from ..types import Displacement
from ..loaders.path_images import PathImages

TCZYX_Tuple = Tuple[int, int, int, int, int]


class ExportChannelType(IntEnum):
    BEADS = 0
    CELLS = 1
    DEFORMATIONS = 2
    RAW_DEFS = 3
    U = 4
    V = 5
    SN = 6


ECT = ExportChannelType


class TiffResultsExporter:
    """
    This class uses the generated images and displacements (u, v) to create deformation heatmaps
    with cell max-projection overlays
    """
    cell_images: PathImages

    def __init__(self, displacement_list: List[Displacement], bead_images: PathImages, cell_images: PathImages,
                 source_name: Path):
        """
        :param displacement_list: List of displacement objects for each pair of images
        :param bead_images: A PathImages(List) of bead images
        :param cell_images: A PathImages(List) of cell max projection images
        :param source_name: The Path to the source filename
        """
        self.disps = displacement_list
        self.bead_images = bead_images
        self.cell_images = cell_images
        self.source_fname = source_name
        self.over_images = PathImages()
        self.min_data_value = 2250  # these are hardcoded for now based on histograms of the deformation data
        self.max_data_value = 4750
        self.length = len(bead_images)
        self.channel_names = ('beads', 'cells', '||deformation||', '||raw defs||', 'u', 'v', 's/n')
        self.output_data = None

    def process(self) -> None:
        """
        This is the function to call from main that will run use the member functions to
        generate a finished deformation movie.
        :return: None
        """
        dims = self.lookup_dimensions()
        self.construct_and_populate_bead_and_cell_images(dims)
        self.populate_deformation_mag(dims)
        self.output_data = np.uint16(self.output_data)

        oname = Path(self.source_fname.parent) / self.source_fname.stem
        oname.with_suffix('.def.tif')
        with OmeTifWriter(oname) as ow:
            ow.save(self.output_data, channel_names=self.channel_names)

    def lookup_dimensions(self) -> TCZYX_Tuple:
        """
        Figure out the right TCZYX shape for the data object
        :return: list/tuple of the dimensions in TCZYX order
        """
        self.cell_images.set_image()
        img = self.cell_images[0]
        dims = (self.length, len(self.channel_names), 1, img.shape[0], img.shape[1])
        return dims

    def construct_and_populate_bead_and_cell_images(self, dims: TCZYX_Tuple):
        """
        Create the internal object memory block and populate it with the Bead and Cell Images
        :param dims: The shape (T, C, Z, Y, X) of the output block
        :return: None
        """
        self.bead_images.set_image()
        self.cell_images.set_image()
        self.output_data = np.uint16(np.zeros(dims))
        for t in range(self.length):
            bead_img = self.bead_images[t]
            cell_img = self.cell_images[t]
            self.output_data[t, ECT.BEADS, 1, :, :] = bead_img[:, :, 0]
            self.output_data[t, ECT.CELLS, 1, :, :] = cell_img[:, :, 0]

    def populate_deformation_mag(self, dims: TCZYX_Tuple) -> None:
        for t in range(1, self.length):
            defdata = self.deformation_mag_to_img(self.disps[t-1], dims)  # our t index is 1 longer than deformations
            defdata *= 65535
            self.output_data[t, ECT.DEFORMATIONS, 1, :, :] = defdata[:, :]
            raw_data = self.deformation_mag_to_img(self.disps[t-1], dims, raw=True)
            raw_data = np.clip(raw_data, 0, 65535)
            self.output_data[t, ECT.RAW_DEFS, 1, :, :] = raw_data

    def deformation_mag_to_img(self, disp: Displacement, dims: TCZYX_Tuple, raw: bool = False) -> np.ndarray:
        dmag = self.arctan_map(disp, raw=raw)
        dy = int((dims[3] - dmag.shape[0]) / 2)
        dx = int((dims[4] - dmag.shape[1]) / 2)
        result = np.zeros((dims[3], dims[4]))
        result[dy:dy + dmag.shape[0], dx:dx + dmag.shape[1]] = dmag
        result = np.flip(result, 0)
        return result

    @classmethod
    def arctan_map(cls, disp: Displacement, raw: bool = False) -> np.ndarray:
        """
        I'm using the arctan to map the deformation intensities into the range (0, 1)
        :param disp: The displacement class
        :param raw: Set to True return the raw values if false scale them from (0, 1)
        :return: the scaled intensity values
        """
        dmag = np.nan_to_num(disp.magnitude_grid, copy=True)
        if not raw:
            dmag = (np.arctan(np.sqrt(3) * (dmag - 3750.0) / 26250.0) + np.pi / 2.0) / np.pi
        return dmag
