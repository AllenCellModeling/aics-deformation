import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from imageio import imwrite
from aicsimageio import AICSImage

from .helpers import raise_a_if_b
from .exception import MinBeadMatchNotMetException, InsufficientTimePointsException
from .loader_abc import LoaderABC
from .path_images import PathImages
from ..types import Displacement


NpImage = np.ndarray  # (Y, X) data image
Dcube = np.ndarray  # (Z, Y, X) data cube
Matrix = np.ndarray
Char = str  # string of length 1
SIFT_Type = cv2.xfeatures2d_SIFT


class CziMoveLoader(LoaderABC):
    """
    This class breaks a single scene movie of cells on beads and finds the frame
    of the `best` bead plane for each time step and generates a corresponding cell
    max_project image for overlay post-piv processing.
    """

    def __init__(self, pathname: Path, bead_channel: Optional[int] = 0, test_data: Optional[np.ndarray] = None):
        """
        Take the starting image with path to enable the construction of sub-folders and frames from the czi movie.
        If test_data is supplied the filepath is not checked it's simply used to compute a folder destination.
        :param pathname: A Path containing the filename and path to get to the
                            czi or tiff image [or a fake if using test_data].
        :param bead_channel: Optional, The channel the beads should be present in
        :param test_data: Optional, For testing. A data cube to enable faster
                            testing as opposed to a large czi/tiff file.
            example `/usr/local/test.czi'
        """
        if test_data is None and not pathname.resolve().is_file():
            raise FileNotFoundError
        self.parent = pathname.parent.resolve()
        self.home = self.parent / pathname.stem
        self.bead_home = self.home / 'beads'  # folder for stabilized bead images
        self.cell_home = self.home / 'cells'  # folder for stabilized cell images
        self.over_home = self.home / 'deformations'  # folder for the deformation / cell overlay images
        self.tmp_bead_home = self.home / 'tmp' / 'beads'  # working folder for 2D image slices / removed after run
        self.tmp_cell_home = self.home / 'tmp' / 'cells'  # working folder for 2D image slices / removed after run
        self.bead_home.mkdir(mode=0o755, parents=True)
        self.cell_home.mkdir(mode=0o755, parents=True)
        self.over_home.mkdir(mode=0o755, parents=True)
        self.tmp_bead_home.mkdir(mode=0o755, parents=True)
        self.tmp_cell_home.mkdir(mode=0o755, parents=True)
        load_this = [x for x in (pathname, test_data) if x is not None]
        self.image = AICSImage(load_this[-1], dims="TCZYX")
        self.bead_channel = bead_channel
        self.bead_images = PathImages()
        self.cell_images = PathImages()
        self.over_images = PathImages()  # Overlay images of deformation and cell
        self.warp_ms = None

    def last_time_frame(self) -> int:
        """
        Get the total number of time-points
        :return: number of time-points
        """
        t_index = self.image_get_index_order('T')
        return self.image.shape[t_index]

    def run_before(self) -> None:
        """
        This command launches all of the pre-processing commands to generate the bead
        plane images as well as the max projection images.
        :return: None
        """
        ti_max = self.last_time_frame()
        dcube = self.image.get_image_data(out_orientation="TZYX", C=self.bead_channel)
        slice_array = self.generate_bead_imgs(ti_max, dcube)
        frame_k_dict = self.cv_find_points(slice_array)
        self.warp_ms = [self.align_warp_and_write_beads(slice_array, frame_k_dict, i) for i in range(len(slice_array))]
        self.generate_projection_imgs(ti_max=ti_max, dcube=dcube)

    def generate_bead_imgs(self, ti_max: int, dcube: Dcube) -> List[NpImage]:
        """
        Find the bead image slice for each time-point and load them into a list
        :return: list of bead images ( 1 per time-point )
        """
        return [self.find_bead_slice(dcube[t_i, :, :, :], t_i) for t_i in range(0, ti_max)]

    def generate_projection_imgs(self, ti_max: int, dcube: Dcube) -> None:
        imgs = [self.max_projection(dcube[ti, :, :, :]) for ti in range(ti_max)]
        w_imgs = [cv2.warpPerspective(imgs[i], self.warp_ms[i], (imgs[0].shape[1], imgs[0].shape[0]))
                  for i in range(len(imgs))]
        f_names = [self.cell_home / f"cells{str(ti).zfill(3)}.jpg" for ti in range(ti_max)]
        [imwrite(str(f_name), img) for f_name, img in zip(f_names, w_imgs)]
        self.cell_images.extend(f_names)

    def image_get_index_order(self, letter: Char) -> int:
        """
        Get the Time index, is it the 1st, 2nd, ... axis of the data object
        :param letter: The Dimension you want the position of in the data object
        :return: The position (int)
        """
        try:
            t_index = self.image.dims.index(letter)
        except ValueError:
            raise InsufficientTimePointsException(
                "Error: image file needs to contain a time axis to do temporal alignment!"
            )
        return t_index

    def find_bead_slice(self, tmp_data: Dcube, time_idx: int) -> NpImage:
        """
        Collapse all the pixels in the (x,y) plain into one value then normalize it to [0,1]
        and then check the maximum value against the maxima of the second derivative.
        :param tmp_data: an (Z, Y, X) data cube for one instance in time
        :param time_idx: time index of the data cube for file name generation
        :return: index of the bead slice or -1 in the case that the algorithm fails
        """
        slice_d = None
        z_data = np.sum(tmp_data, axis=(1, 2))
        z_data = z_data/np.max(z_data)
        dz2 = np.gradient(np.gradient(z_data))
        z_idx = np.argmax(z_data)
        dz2_ind = np.argmin(dz2)
        if z_idx == dz2_ind or dz2[dz2_ind] < 0.0:
            filename = self.tmp_bead_home / f"bead_{str(time_idx).zfill(4)}_z{str(z_idx).zfill(3)}.jpg"
            slice_d = tmp_data[z_idx, :, :]
            imwrite(filename, slice_d)
        return slice_d

    @classmethod  # should these be class methods?
    def cv_find_points_in_slice(cls, sift: SIFT_Type, bead_img: NpImage) -> Dict:
        """
        This function constructs a dictionary of key-points and descriptors that can
        then be used to do a Ransac alignment to correct for camera jitter.
        :param sift: The SIFT object with attached methods for finding key-points
        :param bead_img: A bead image to find key-points in
        :return: A dictionary of key-points and descriptors
        """
        img = cv2.normalize(bead_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        kp, des = sift.detectAndCompute(img, None)
        return {'key_points': kp, 'descriptors': des}

    @classmethod
    def cv_find_points(cls, slice_arr: List) -> List[Dict]:
        """
        Use opencv to find the beads in the slice
        :param slice_arr: a list of the bead slices to be aligned
        :return: a list of dictionaries, each dictionary containing image key points and descriptors
        """
        sift = cv2.xfeatures2d.SIFT_create()
        return [cls.cv_find_points_in_slice(sift, slice_arr[f_i]) for f_i in range(len(slice_arr))]

    def align_warp_and_write_beads(self, bead_imgs: List, k_frames: List, idx: int) -> Matrix:
        """
        Calculate the transformation matrix (m) warp the image at index idx and write the stabilized
        image to the bead_home folder.
        :param bead_imgs: list of bead images to be aligned
        :param k_frames: list of dictionaries containing points corresponding to images
        :param idx: the index of the image to be aligned against the reference time-point (0)
        :return: the transformation matrix(3x3) (m) so the max projection images can be similarly aligned
        """
        m = self.align_pair(k_frames[idx], k_frames[0])
        w_img = cv2.warpPerspective(bead_imgs[idx], m, (bead_imgs[0].shape[1], bead_imgs[0].shape[0]))
        f_name = self.bead_home / f"beads{str(idx).zfill(3)}.jpg"
        self.bead_images.append(f_name)
        imwrite(f_name, w_img)
        return m

    @classmethod
    def align_pair(cls, a: Dict, b: Dict) -> Matrix:
        """
        Find the matrix transform between 2 bead images (Homography) and then apply
        return it so it can be used to warp the appropriate images (the bead slice and the max projection)
        :param a: dictionary of key-points and descriptors
        :param b: reference dictionary of key-points and distances
        :return: transformation matrix from index to 0 reference frame
        """
        min_match_count = 50
        flann_index_kdtree = 0
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(a['descriptors'], b['descriptors'], k=2)
        #
        good = [m for m, n in matches if m.distance < 0.7*n.distance]
        raise_a_if_b(MinBeadMatchNotMetException(len(good), min_match_count), len(good) < min_match_count)

        src_key_pts = a['key_points']
        dst_key_pts = b['key_points']
        src_pts = np.float32([src_key_pts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([dst_key_pts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return matrix

    @classmethod
    def max_projection(cls, zyx_data: Dcube) -> NpImage:
        """
        Compute the max projection of the 3d data along the Z axis
        :param zyx_data: A data cube with Dimensions (Z, Y, X)
        :return: A flattened 2D Image
        """
        return np.max(zyx_data, 0)
