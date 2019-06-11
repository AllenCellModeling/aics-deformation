import pytest
import numpy as np
import pickle
import aicsdeformation.finishers.tiff_exporter as te
from aicsdeformation.loaders.path_images import PathImages
from aicsdeformation.types import Displacement


@pytest.fixture
def construct_tre(data_dir):
    cell_path = data_dir / "overlay/cells"
    bead_path = data_dir / "overlay/beads"
    cell_imgs = PathImages()
    bead_imgs = PathImages()
    [cell_imgs.append(img) for img in cell_path.glob("*.png")]
    [bead_imgs.append(img) for img in bead_path.glob("*.png")]

    with open(str(data_dir / "overlay" / "xarr.pkl"), "rb") as fp:
        xv = pickle.load(fp)
    with open(str(data_dir / "overlay" / "yarr.pkl"), "rb") as fp:
        yv = pickle.load(fp)

    x = np.zeros((1293, 1893))
    y = np.zeros((1293, 1893))
    for i in range(1893):  # turn the vector into a grid
        y[:, i] = yv[:]
    for j in range(1293):  # turn the vector into a grid
        x[j, :] = xv[:]

    with open(str(data_dir / "overlay" / "uarr.pkl"), "rb") as fp:
        u = pickle.load(fp)
    with open(str(data_dir / "overlay" / "varr.pkl"), "rb") as fp:
        v = pickle.load(fp)
    with open(str(data_dir / "overlay" / "s2n.pkl"), "rb") as fp:
        sn = pickle.load(fp)
    with open(str(data_dir / "overlay" / "mask.pkl"), "rb") as fp:
        mask = pickle.load(fp)

    disp = Displacement(u=u, v=v, sig2noise=sn, x=x, y=y, mask=mask)
    disps = [disp, disp, disp, disp]

    mock_czi_path = data_dir / "output.czi"

    tre = te.TiffResultsExporter(displacement_list=disps, bead_images=bead_imgs,
                                 cell_images=cell_imgs, source_name=mock_czi_path)
    return tre


def test_data_shape(construct_tre):
    tre = construct_tre
    assert len(tre.bead_images) == len(tre.cell_images)
    assert len(tre.bead_images) == 1+len(tre.disps)


def test_tiff_results_exporter(construct_tre):
    tre = construct_tre
    tre.process()
    assert True


def test_mapping_two():
    x = te.TiffResultsExporter.scale_value(1000000)
    assert(x > 64500)


def test_mapping_three():
    x = te.TiffResultsExporter.scale_value(-1000000)
    assert(x < 500)


def test_mapping_four():
    x = te.TiffResultsExporter.scale_value(0)
    assert(x < 500)
