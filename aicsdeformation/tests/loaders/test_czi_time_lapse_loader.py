import pytest

import cv2
import numpy as np
import random

from aicsdeformation.loaders.czi_time_lapse_loader import CziTimeLapseLoader, CellChannelType

MOCK_BEAD_PLANE = np.random.rand(5, 2, 10, 100, 100)
MOCK_BEAD_PLANE[:, :, 3, :, :] = 255*np.ones((5, 2, 100, 100))
MOCK_BEAD_PLANE[:, :, 0:2, :, :] *= 125
MOCK_BEAD_PLANE[:, : 4:, :, :] *= 125
MOCK_BEAD_PLANE = np.uint8(MOCK_BEAD_PLANE)


@pytest.fixture
def instantiate_czi_movie_loader(data_dir):
    from pathlib import Path
    from shutil import rmtree
    fname = data_dir / '20180924_M01_001-Scene-01-P1.czi'
    working_folder = fname.parent / Path(fname.stem)
    if working_folder.exists():  # if the folder exists remove it before it's passed to the class to construct
        rmtree(working_folder)
    cziml = CziTimeLapseLoader(pathname=fname, test_data=MOCK_BEAD_PLANE)
    return cziml, fname.resolve()


@pytest.fixture
def get_czi_class_and_time(data_dir):
    from pathlib import Path
    from shutil import rmtree
    fname = data_dir / '20180924_M01_001-Scene-01-P1.czi'
    working_folder = fname.parent / Path(fname.stem)
    if working_folder.exists():  # if the folder exists remove it before it's passed to the class to construct
        rmtree(working_folder)
    cziml = CziTimeLapseLoader(pathname=fname, test_data=MOCK_BEAD_PLANE)
    return cziml, MOCK_BEAD_PLANE.shape[0]


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
    assert len(cziml) == time


@pytest.mark.parametrize("f_name, data, arg_dict, expected",[
    pytest.param("test.czi", MOCK_BEAD_PLANE[0, 0, :, :, :], {'out_orientation': "ZYX", 'T': 0, 'C': 0}, 3,
                 marks=pytest.mark.raises(exception=ValueError)),
    ("test.czi", MOCK_BEAD_PLANE, {'out_orientation': "ZYX"}, 3)
])
def test_gradient_length(data_dir, f_name, data, arg_dict, expected):
    p_name = data_dir / f_name
    cziml = CziTimeLapseLoader(pathname=p_name, test_data=data)
    l_data = cziml.image.get_image_data(**arg_dict)
    d_slice = cziml.find_bead_slice(l_data, 0, rescale=False)
    np.testing.assert_array_equal(l_data[expected, :, :], d_slice)


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
    d_slice *= 255  # 65535
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
