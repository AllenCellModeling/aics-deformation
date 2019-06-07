import pytest
from pathlib import Path
import pickle
import aicsdeformation.finishers.tiff_exporter as te
from aicsdeformation.loaders.path_images import PathImages


@pytest.fixture
def construct_tre(data_dir):
    cell_path = data_dir / "overlay/cells"
    bead_path = data_dir / "overlay/beads"
    cell_imgs = PathImages()
    bead_imgs = PathImages()
    [cell_imgs.append(img) for img in cell_path.glob("*.png")]
    [bead_imgs.append(img) for img in bead_path.glob("*.png")]
    defs_pkl = data_dir / "overlay/five.pkl"
    with open(str(defs_pkl), "rb") as fp:
        disps = pickle.load(fp)
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
