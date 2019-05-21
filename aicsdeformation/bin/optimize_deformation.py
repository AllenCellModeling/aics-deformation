from argparse import ArgumentParser
import logging
from pathlib import Path
import pickle

from aicsdeformation.loaders.czi_time_lapse_loader import CziTimeLapseLoader
from aicsdeformation.processing import grid_search_displacements

log = logging.getLogger("Deformations")
logging.captureWarnings(True)


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--image_path", help="path to image (czi or tiff)", required=True)
    parser.add_argument("-b", "--bead_channel", help="the channel in the image containing bead data (default = 1)",
                        type=int, default=1)
    parser.add_argument("-c", "--cell_channel", help="the channel in the image containing the cell / "
                                                     "structure (default = 0)", type=int, default=0)

    js_args = parser.parse_args()

    # process the image file into it's required input folders and image files
    img = CziTimeLapseLoader(pathname=Path(js_args.image_path), bead_channel=js_args.bead_channel,
                             cell_channel=js_args.cell_channel)
    img.process()

    best, all_displacements = grid_search_displacements(frame_a=img.bead_images[2], frame_b=img.bead_images[3],
                                                        window_size_min=8, overlap_min=2, search_area_size_min=8,
                                                        window_size_max=32, search_area_size_max=32)

    print(best.parameters)

    # serialize deformations
    f_path = img.over_home / "best_params.pkl"
    with open(f_path, 'wb') as fp:
        pickle.dump(best.parameters, fp)


if __name__ == '__main__':
    main()
