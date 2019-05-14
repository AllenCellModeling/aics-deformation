from argparse import ArgumentParser
from pathlib import Path
import pickle

from aicsdeformation.loaders.czi_time_lapse_loader import CziTimeLapseLoader
from aicsdeformation.aicsdeformation import AICSDeformation
from aicsdeformation.finishers.overlay_generator import OverlayGenerator


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--image_path", help="path to image (czi or tiff)", required=True)

    js_args = parser.parse_args()

    # process the image file into it's required input folders and image files
    img = CziTimeLapseLoader(pathname=Path(js_args.image_path))
    img.run_before()

    # call Jackson's code here to launch openPIV
    aics_def = AICSDeformation(frames=img.bead_images)

    # calculate deformations
    disp_list = aics_def.generate_displacements()

    # serialize deformations
    f_path = img.over_home / "deformations.pkl"
    with open(f_path, 'wb') as fp:
        pickle.dump(disp_list, fp)

    # create deformation / cell image overlays
    fin = OverlayGenerator(disp_list, img.cell_images, img.over_home)
    fin.finish()


if __name__ == '__main__':
    main()
