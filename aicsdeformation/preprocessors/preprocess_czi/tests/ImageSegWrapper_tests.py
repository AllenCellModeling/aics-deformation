import pytest
import piv_image_spilitter.ImageSegWrapper as isw


@pytest.fixture
def aics_im_obj():
    fname = '../../Data/20180924_M01_001-Scene-01-P1.czi'
    iw = isw.ImgSegWrapper(fname)
    return iw


def test_time_length(aics_im_obj):
    t_length = aics_im_obj.time_range()
    assert t_length == 150


def test_gradient_length_zero(aics_im_obj):
    l_data = aics_im_obj.get_image_data(out_orientation="ZYX", T=0, C=0)
    idx = aics_im_obj.find_beed_slice(l_data)
    assert idx == 5


def test_gradient_length(aics_im_obj):
    l_data = aics_im_obj.get_image_data(out_orientation="ZYX", T=1, C=1)
    idx = aics_im_obj.find_beed_slice(l_data)
    assert idx == 6


def test_create_files(aics_im_obj):
    aics_im_obj.perform_alignment()
    assert True



