from argparse import ArgumentParser
import logging
from pathlib import Path

from aicsdeformation.loaders.czi_time_lapse_loader import CziTimeLapseLoader, CellChannelType
from aicsdeformation.aicsdeformation import AICSDeformation
from aicsdeformation.finishers.tiff_exporter import TiffResultsExporter

log = logging.getLogger("Deformations")
logging.captureWarnings(True)


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--image_path", help="path to image (czi or tiff)", required=True)
    parser.add_argument("-b", "--bead_channel", help="the channel in the image containing bead data (default = 1)",
                        type=int, default=1)
    parser.add_argument("-c", "--cell_channel", help="the channel in the image containing the cell / "
                                                     "structure (default = 0)", type=int, default=0)
    parser.add_argument("--cell_channel_type", help="BRIGHTFIELD, GFP", type=str, default="BRIGHTFIELD")

    js_args = parser.parse_args()

    channel_type = CellChannelType.BRIGHT_FIELD
    if js_args.cell_channel_type == "GFP":
        channel_type = CellChannelType.GFP

    # process the image file into it's required input folders and image files
    img = CziTimeLapseLoader(pathname=Path(js_args.image_path), bead_channel=js_args.bead_channel,
                             cell_channel=js_args.cell_channel, cell_channel_type=channel_type)
    img.process()

    # call Jackson's code here to launch openPIV
    aics_def = AICSDeformation(frames=img.bead_images)

    # calculate deformations
    # disp_list = aics_def.generate_displacements(window_size=16, overlap=4, search_area_size=16, dt=0.002)
    disp_list = aics_def.generate_displacements(window_size=8, overlap=7, search_area_size=8, dt=0.002)

    # create deformation / cell image overlays
    fin = TiffResultsExporter(displacement_list=disp_list, bead_images=img.bead_images,
                              cell_images=img.cell_images, source_name=Path(js_args.image_path))
    fin.process()


if __name__ == '__main__':
    main()
