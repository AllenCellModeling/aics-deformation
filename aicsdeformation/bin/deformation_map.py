from argparse import ArgumentParser
from pathlib import Path
import pickle

from aicsdeformation.loaders.czi_movie_loader import CziMoveLoader
from aicsdeformation.aicsdeformation import AICSDeformation
from aicsdeformation.finishers.cell_deformation_overlays import OverlayFinisher


def main():
    parser = ArgumentParser()
    parser.add_argument("-ip", "--image_path", help="path to image (czi or tiff)", required=True)

    js_args = parser.parse_args()

    # process the image file into it's required input folders and image files
    img = CziMoveLoader(pathname=Path(js_args.image_path))
    img.run_before()

    # call Jackson's code here to launch openPIV
    aics_def = AICSDeformation(frames=img.bead_images)

    # calculate deformations
    disp_list = aics_def.generate_displacements()

    print(disp_list[0])
    # serialize deformations
    f_path = img.over_home / "deformations.pkl"
    with open(f_path, 'wb') as fp:
        pickle.dump(disp_list, fp)

    # create deformation / cell image overlays
    fin = OverlayFinisher(disp_list, img.cell_images, img.over_home)
    fin.finish()


if __name__ == '__main__':
    main()
