from enum import IntEnum
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

from aicsimageio.omeTifWriter import OmeTifWriter

from ..types import Displacement
from ..loaders.path_images import PathImages

TCZYX_Tuple = Tuple[int, int, int, int, int]

IMG_MAX = 65535


class ExportChannelType(IntEnum):
    BEADS = 0
    CELLS = 1
    DEFORMATIONS = 2  # Scaled
    RAW_DEFS = 3      # Raw values
    U_P = 4           # U >=0
    U_M = 5           # |U| for U < 0
    V_P = 6           # V >=0
    V_M = 7           # |V| for V < 0
    SN = 8            # S/N


ECT = ExportChannelType


class TiffResultsExporter:
    """
    This class uses the generated images and displacements (u, v) to create deformation heatmaps
    with cell max-projection overlays
    """
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
        self.length = len(bead_images)
        self.channel_names = ('beads', 'cells', '||deformation||', '||raw defs||')
        # '||u+||', '||u-||', '||v+||', '||v-||', 's/n') to add back in later
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
        oname = oname.with_suffix('.def.tif')
        with OmeTifWriter(oname, overwrite_file=True) as ow:
            data = np.transpose(self.output_data, (0, 2, 1, 3, 4))
            ow.save(data=data, channel_names=self.channel_names)

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
            self.output_data[t, ECT.BEADS, 0, :, :] = bead_img[:, :]
            self.output_data[t, ECT.CELLS, 0, :, :] = cell_img[:, :]

    def populate_deformation_mag(self, dims: TCZYX_Tuple) -> None:
        for t in range(1, self.length):
            # scaled deformations
            frame = self.deformation_mag_to_img(self.disps[t-1].magnitude, dims)  # our t index is 1 + deformations
            self.output_data[t, ECT.DEFORMATIONS, 0, :, :] = frame[:, :]
            # raw deformations
            frame = self.deformation_mag_to_img(self.disps[t-1].magnitude, dims, raw=True)
            frame = np.clip(frame, 0, IMG_MAX)
            self.output_data[t, ECT.RAW_DEFS, 0, :, :] = frame
            # # U_P
            # frame = np.clip(self.disps[t-1].u, a_min=0, a_max=IMG_MAX)
            # frame = self.deformation_mag_to_img(frame, dims, raw=True)
            # self.output_data[t, ECT.U_P, 0, :, :] = frame
            # # U_M
            # frame = -1.0 * np.clip(self.disps[t-1].u, a_min=-IMG_MAX, a_max=0)
            # frame = self.deformation_mag_to_img(frame, dims, raw=True)
            # self.output_data[t, ECT.U_M, 0, :, :] = frame
            # # V_P
            # frame = np.clip(self.disps[t - 1].v, a_min=0, a_max=IMG_MAX)
            # frame = self.deformation_mag_to_img(frame, dims, raw=True)
            # self.output_data[t, ECT.V_P, 0, :, :] = frame
            # # V_M
            # frame = -1.0 * np.clip(self.disps[t - 1].v, a_min=-IMG_MAX, a_max=0)
            # frame = self.deformation_mag_to_img(frame, dims, raw=True)
            # self.output_data[t, ECT.V_M, 0, :, :] = frame
            # # S/N
            # frame = np.clip(self.disps[t - 1].sig2noise, a_min=0, a_max=IMG_MAX)
            # frame = self.deformation_mag_to_img(frame, dims, raw=True)
            # self.output_data[t, ECT.SN, 0, :, :] = frame

    def deformation_mag_to_img(self, grid_data: np.ndarray, dims: TCZYX_Tuple, raw: bool = False) -> np.ndarray:
        dmag = self.arctan_map(grid_data, raw=raw)
        dy = int((dims[3] - dmag.shape[0]) / 2)
        dx = int((dims[4] - dmag.shape[1]) / 2)
        result = np.zeros((dims[3], dims[4]))
        result[dy:dy + dmag.shape[0], dx:dx + dmag.shape[1]] = dmag
        return result

    @classmethod
    def arctan_map(cls, d_grid: np.ndarray, raw: bool = False) -> np.ndarray:
        """
        I'm using the arctan to map the deformation intensities into the range (0, 1)
        :param d_grid: The numpy array of values to be scaled
        :param raw: Set to True return the raw values if false scale them from (0, 1)
        :return: the scaled intensity values
        """
        dmag = np.nan_to_num(d_grid, copy=True)
        if not raw:
            dmag = cls.scale_value(dmag)
        return dmag

    @classmethod
    def scale_value(cls, d_val: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        ans = ((np.arctan(np.sqrt(3) * (d_val - 3750.0) / 26250.0) + (np.pi / 2.0)) / np.pi)
        ans = (ans - 0.42)/(1.0 - 0.42)
        return IMG_MAX*ans
