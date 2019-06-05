import pytest

import cv2
import numpy as np
import random

from aicsdeformation.loaders.czi_time_lapse_loader import CziTimeLapseLoader, CellChannelType


class ImgContainer(object):
    def __init__(self, channels: int = 5, dims: str = "TCZYX"):
        self.input_shape = random.sample(range(1, 10), channels)
        self.input_shape[0] += 2  # require a minimum of 2 time-steps
        self.input_shape[1] += 3  # require a minimum of 3 Channels
        self.input_shape[2] = 9   # set the size of Z to 9 slices
        self.stack = np.zeros(self.input_shape)
        x_mag = self.input_shape[-1]
        y_mag = self.input_shape[-2]
        # set T=0, C=0 Z=4 to a bead slice (non-zeros)
        beads = self.stack[0, 1, 4, :, :]
        beads2 = self.stack[0, 1, 5, :, :]
        samples = int(x_mag*y_mag/5)
        xs = [random.randrange(0, x_mag) for _ in range(samples)]
        ys = [random.randrange(0, y_mag) for _ in range(samples)]
        for y_i, x_i in zip(ys, xs):
            beads[y_i, x_i] = beads2[y_i, x_i] = 1
        self.dims = dims
        self.order = {c: i for i, c in enumerate(dims)}  # {'T': 0, 'C': 1, 'Z': 2, 'Y': 3, 'X': 4}

    def n_of_time_points(self):
        return self.input_shape[0]


@pytest.fixture
def example_img5():
    return ImgContainer()


@pytest.fixture
def instantiate_czi_movie_loader(data_dir):
    from pathlib import Path
    from shutil import rmtree
    fname = data_dir / '20180924_M01_001-Scene-01-P1.czi'
    working_folder = fname.parent / Path(fname.stem)
    if working_folder.exists():  # if the folder exists remove it before it's passed to the class to construct
        rmtree(working_folder)
    img_obj = ImgContainer()
    cziml = CziTimeLapseLoader(pathname=fname, test_data=img_obj.stack)
    return cziml, fname.resolve()


# @pytest.fixture
# def instantiate_czi_movie_loader_two(data_dir):
#     from pathlib import Path
#     from shutil import rmtree
#     fname = data_dir / '20190425_S08_001-04-Scene-3-P4-B03.czi'
#     working_folder = fname.parent / Path(fname.stem)
#     if working_folder.exists():  # if the folder exists remove it before it's passed to the class to construct
#         rmtree(working_folder)
#     img_obj = ImgContainer()
#     cziml = CziTimeLapseLoader(pathname=fname)
#     return cziml, fname.resolve()


@pytest.fixture
def get_czi_class_and_time(data_dir):
    from pathlib import Path
    from shutil import rmtree
    fname = data_dir / '20180924_M01_001-Scene-01-P1.czi'
    working_folder = fname.parent / Path(fname.stem)
    if working_folder.exists():  # if the folder exists remove it before it's passed to the class to construct
        rmtree(working_folder)
    img_obj = ImgContainer()
    cziml = CziTimeLapseLoader(pathname=fname, test_data=img_obj.stack)
    return cziml, img_obj.n_of_time_points()


def test_parent(instantiate_czi_movie_loader):
    cziml, fname = instantiate_czi_movie_loader
    pth = fname.parent
    assert pth == cziml.parent


def test_home(instantiate_czi_movie_loader):
    cziml, fname = instantiate_czi_movie_loader
    pth = fname.parent / '20180924_M01_001-Scene-01-P1'
    assert pth == cziml.home


def test_bead(instantiate_czi_movie_loader):
    cziml, fname = instantiate_czi_movie_loader
    pth = fname.parent / '20180924_M01_001-Scene-01-P1/beads'
    assert pth == cziml.bead_home


def test_cells(instantiate_czi_movie_loader):
    cziml, fname = instantiate_czi_movie_loader
    pth = fname.parent / '20180924_M01_001-Scene-01-P1/cells'
    assert pth == cziml.cell_home


def test_time_length(get_czi_class_and_time):
    cziml, time = get_czi_class_and_time
    t_length = len(cziml)
    assert t_length == time


def test_gradient_length_zero(instantiate_czi_movie_loader):
    cziml, fname = instantiate_czi_movie_loader
    l_data = cziml.image.get_image_data(out_orientation="ZYX", T=0, C=0)
    d_slice = cziml.find_bead_slice(l_data, 0)
    y_max, x_max = l_data[5, :, :].shape
    for y_i in range(y_max):
        for x_i in range(x_max):
            assert (l_data[5, y_i, x_i] == d_slice[y_i, x_i])


def test_gradient_length(instantiate_czi_movie_loader):
    cziml, fname = instantiate_czi_movie_loader
    l_data = cziml.image.get_image_data(out_orientation="ZYX", T=1, C=1)
    d_slice = cziml.find_bead_slice(l_data, 1)
    y_max, x_max = l_data[6, :, :].shape
    for y_i in range(y_max):
        for x_i in range(x_max):
            assert(l_data[6, y_i, x_i] == d_slice[y_i, x_i])


def test_cv2_imwrite():
    nparr = np.zeros((100, 100))
    nparr = np.uint16(nparr)
    cv2.imwrite("test.png", nparr)


def test_max_projection():
    dcube = np.zeros((100, 100, 100))  # make a ZYX data cube
    d_slice = np.zeros((100, 100))
    n_of_rand = int(100*100/10)  # 10% of a 2D image
    zs = [random.randint(0, 99) for i in range(n_of_rand)]
    ys = [random.randint(0, 99) for i in range(n_of_rand)]
    xs = [random.randint(0, 99) for i in range(n_of_rand)]
    for z_i, y_i, x_i in zip(zs, ys, xs):
        dcube[z_i, y_i, x_i] = d_slice[y_i, x_i] = 1

    lb = CellChannelType.BRIGHT_FIELD
    d_test = CziTimeLapseLoader.max_projection(dcube, lb)
    d_slice *= 65535
    d_slice = np.uint16(d_slice)
    for y_i in range(0, 99):
        for x_i in range(0, 99):
            assert d_slice[y_i, x_i] == d_test[y_i, x_i]


# def test_problematic_czi(instantiate_czi_movie_loader_two):
#     cziml, fname = instantiate_czi_movie_loader_two
#     l_data = cziml.image.get_image_data(out_orientation="ZYX", T=0, C=1)
#     d_slice = cziml.find_bead_slice(l_data, 0)
#     y_max, x_max = l_data[6, :, :].shape
#     for y_i in range(y_max):
#         for x_i in range(x_max):
#             assert (l_data[6, y_i, x_i] == d_slice[y_i, x_i])
