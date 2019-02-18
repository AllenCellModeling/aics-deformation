import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print(plt.get_backend())

import openpiv.tools
import openpiv.process
import openpiv.scaling
import openpiv.validation
import openpiv.filters
import numpy as np
import pandas as pd

import os
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial

import glob

print(plt.get_backend())


# Class from Forrest for doing multi-threaded processing; used in calculate_displacements function to speed up
# processing across multiple cores.
class WithPool(Pool):
    """pathos ProcessingPool with functioning __exit__ call
    Parameters
    ----------
    *args
        variable length argument list matching input
        to pathos.multiprocessing.Pool
    **kwargs
        keyword argument input matching pathos.multiprocessing.Pool
    Examples
    --------
    >>> with WithPool(number_processes) as pool:
    >>>     pool.map(myfunc, myInput)
    """

    def __init__(self, *args, **kwargs):
        super(WithPool, self).__init__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        super(WithPool, self)._clear()


class PIVbatcher(object):

    def __init__(self, path, window, overlap, search_area, opath):
        """
        Set up the control member variables needed by PIVbatcher.

        Example
        -------
        PIVbatcher(
                '/AssayDev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
                16, 4, 18, '/Projects/PIV/res2'
            )

        Parameters
        ----------
        path: string
            Filepath to the folder containing the beed and cell image folders
        window: int
            the size of the interrogation window in frame A, how much of the image to use at a time
        overlap: int
            the overlap in pixels between adjacent windows in frame A
        search_area: int
            the size of the window in frame B to that the algorithm can use to attempt to align frame A.
            must be greater or equal to window
        opath: string
            Filepath to write the outputs to.

        Returns
        -------
        PIVbatcher object

        Errors
        ------
        pure assignment the only error possible from the constructor is if a value isn't assignable

        """
        self.framesToSearch = 20  # was 20
        self.path = path
        self.window_size = window
        self.overlap_size = overlap
        self.dt = 0.003
        self.search_area = search_area
        self.s2n_method = 'peak2peak'
        self.outlier_method = 'localmean'
        self.s2n_thresh = 1.3  # was 2.5
        self.output_path = opath
        self.n_of_threads = 16

    def get_beads_dirs(self, item):
        """
        get list of all beads directories within item that end in _beads
        Parameters
        ----------
        item: string
            the starting directory to search under

        Returns
        -------
        bead_dirs: list
            containing the directories found
        """
        beads_dirs = []
        for root, dirs, files in os.walk(item, followlinks=True):
            for name in dirs:
                if name.endswith('_beads'):
                    beads_dirs.append(os.path.join(root, name))
        return beads_dirs

    def get_cells_difs(self, item):
        """
         get list of all cell time step images directories within item that end in _cells
         Parameters
         ----------
         item: string
             the starting directory to search under

         Returns
         -------
         cells_dirs: list
             containing the directories found
         """
        cells_dirs = []
        for root, dirs, files in os.walk(item, followlinks=True):
            for name in dirs:
                if name.endswith('_cells'):
                    cells_dirs.append(os.path.join(root, name))
        return cells_dirs

    # function to calculate displacement between two images
    def calc_displacements(self, fnx, fny):
        """
        calculate displacement between two images. This function does the heavy lifting for
        the class

        Parameters
        ----------
        fnx: Image A containing beads
        fny: Image B containing beads

        Returns
        -------
        (x, y): coordinates of the centers of the interrogation windows in Image A
        (u, v): the displacement vectors between Image A and Image B
        mask: the points in the grid where the signal to noise was above the defined threshold and nan where it wasn't
            (linear interpolation was used to fill in those points from the surrounding neighbors)
        """
        frame_a, frame_b = openpiv.tools.imread(fnx), openpiv.tools.imread(fny)
        u, v, sig2noise = openpiv.process.extended_search_area_piv(frame_a.astype(np.int32),
                                                                   frame_b.astype(np.int32),
                                                                   window_size=self.window_size,
                                                                   overlap=self.overlap_size,
                                                                   dt=self.dt,
                                                                   search_area_size=self.search_area,
                                                                   sig2noise_method=self.s2n_method)
        x, y = openpiv.process.get_coordinates(image_size=frame_a.shape, window_size=self.window_size,
                                               overlap=self.overlap_size)
        u, v, mask = openpiv.validation.sig2noise_val(u, v, sig2noise, threshold=self.s2n_thresh)
        u, v = openpiv.filters.replace_outliers(u, v, method=self.outlier_method, max_iter=10, kernel_size=2)
        x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor=96.52)
        return x, y, u, v, mask

    # nested for loops to loop through each folder of beads and calcalate displacements between images
    def process_file(self, new_dir, file_n):
        filepath1, filepath2, n = file_n
        txt_filename = new_dir + os.sep + 'bead_displacement_' + 'im' + str(n) + '.csv'
        x, y, u, v, mask = self.calc_displacements(filepath1, filepath2)
        openpiv.tools.save(x, y, u, v, mask, txt_filename)
        return txt_filename

    # function to pass files and saving directory to the process file function, which then passes those files to the
    # calculate_displacements function; allows for implementation of multi-threading
    def calculate_displacements_for_folder(self, folders):
        for bead_folder in folders:
            files_list = []
            files = [f for f in os.listdir(bead_folder) if f.endswith('tif')]
            files.sort()
            for item in files:
                filepath = os.path.join(bead_folder, item)
                files_list.append(filepath)
            new_dir = files_list[1][:-4] + '_between_frames'
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            file_tuples = zip(files_list[0:-1], files_list[1:], range(len(files_list)))
            with WithPool(self.n_of_threads) as p:
                results = p.map(partial(self.process_file, new_dir), file_tuples)

    # Block of functions to find the least-deformed frame within the first 20 frames, and then calculate displacments
    # relative to that frame

    def get_index_of_still_frame(self, beads_filelist):
        diff_list = np.zeros(self.framesToSearch)
        for i in range(self.framesToSearch):
            fnx, fny = beads_filelist[i], beads_filelist[i + 1]
            frame_a, frame_b = openpiv.tools.imread(fnx), openpiv.tools.imread(fny)
            diff_list[i] = np.sum(np.abs(frame_b - frame_a))
            still_frame = np.argmin(diff_list)
        return still_frame

    def process_file_still_frame(self, new_dir, file_n):
        filepath1, filepath2, n = file_n
        txt_filename = new_dir + os.sep + 'bead_displacement_to_still_frame' + 'im' + str(n) + '.csv'
        x, y, u, v, mask = self.calc_displacements(filepath1, filepath2)
        openpiv.tools.save(x, y, u, v, mask, txt_filename)
        return txt_filename

    def calculate_displacements_for_folder_still_frame(self, folders):
        for bead_folder in folders:
            files_list = []
            files = [f for f in os.listdir(bead_folder) if f.endswith('tif')]
            files.sort()
            for item in files:
                filepath = os.path.join(bead_folder, item)
                files_list.append(filepath)
            still_frame = self.get_index_of_still_frame(files_list)
            new_dir = files_list[1][:-4] + '_to_still_frame'
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            fixed_frame_list = [files_list[still_frame] for i in range(len(files_list) - 1)]
            file_tuples = zip(fixed_frame_list, files_list[1:], range(len(files_list)))
            with WithPool(self.n_of_threads) as p:
                results = p.map(partial(self.process_file_still_frame, new_dir), file_tuples)

    # Function to get cell images and displacements to make heatmaps
    def get_cellims_displacements(self, cells_directory, beads_directory):
        cell_im_filelist = []
        all_displacements_list = []
        for folder in beads_directory:
            displacements_filelist = glob.glob(folder + '/*between_frames/*.csv')
            displacements_filelist.sort()
            all_displacements_list.append(displacements_filelist)
        for image in cells_directory:
            folder_cell_list = glob.glob(image + '/*.tif')
            folder_cell_list.sort()
            cell_im_filelist.append(folder_cell_list)
        return cell_im_filelist, all_displacements_list

    # Function to get bead images and displacements (to make heatmaps overlaid with beads instead of cells)
    def get_beadsims_displacements(self, beads_directory):
        bead_im_filelist = []
        all_displacements_list = []
        for folder in beads_directory:
            displacements_filelist = glob.glob(folder + '/*between_frames/*.csv')
            displacements_filelist.sort()
            all_displacements_list.append(displacements_filelist)
        for image in beads_directory:
            folder_cell_list = glob.glob(image + '/*.tif')
            folder_cell_list.sort()
            bead_im_filelist.append(folder_cell_list)
        return bead_im_filelist, all_displacements_list

    # Function to pair up images and displacements to make heatmaps
    def zip_up_ims_and_displacements(self, im_filelist, all_displacements_list):
        csv_im_pair = []
        im_filelist.sort()
        all_displacements_list.sort()
        for i in range(len(all_displacements_list)):
            #         print len(all_displacements_list),len(im_filelist)
            file_tuples = zip(im_filelist[i], all_displacements_list[i])
            for im, csv in file_tuples:
                csv_im_pair.append([im, csv])
        return csv_im_pair

    # Function to take list of paired csvs and images (overlay parameters optimzied for cells)
    # and make heatmap overlays with them
    def make_heatmap_cell_overlays(self, csv_im_pair_list):
        column_names = ['x', 'y', 'u', 'v', 'mask']
        for im, csv in csv_im_pair_list:
            output_path = os.path.dirname(im) + os.sep + 'heatmap_rel' + os.sep
            df = pd.read_csv(csv, '\t', header=None, names=column_names)
            df['m'] = np.sqrt(df['u'] ** 2 + df['v'] ** 2)
            Z1 = df.pivot('y', 'x', 'm').values
            mask = df.pivot('y', 'x', 'mask')
            #Z1 *= 1 - mask
            xmin, xmax, ymin, ymax = df.x.min(), df.x.max(), df.y.min(), df.y.max()
            extent = xmax, xmin, ymax, ymin
            fig = plt.figure(frameon=False)
            im1 = plt.imshow(Z1, cmap=plt.cm.magma, alpha=.9, interpolation='nearest',
                             extent=extent, vmin=0, vmax=15, origin='lower')
            Z2 = openpiv.tools.imread(im)
            im2 = plt.imshow(Z2, cmap=plt.cm.gray, alpha=.3, interpolation='bilinear',
                             extent=extent)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.tight_layout(True)
            plt.axis('off')
            fig.savefig(output_path + im.split('/')[-1].split('.')[0] + "_heatmap_rel_with_cells.png",
                        bbox_inches='tight')
            fig.clear()

    # Function to take list of paired csvs and images (overlay parameters optimzied for beads)
    # and make heatmap overlays with them

    def make_heatmap_bead_overlays(self, csv_im_pair_list):
        column_names = ['x', 'y', 'u', 'v', 'mask']
        for im, csv in csv_im_pair_list:
            output_path = os.path.dirname(im) + os.sep + 'heatmap_rel_over_beadz' + os.sep
            df = pd.read_csv(csv, '\t', header=None, names=column_names)
            df['m'] = np.sqrt(df['u'] ** 2 + df['v'] ** 2)
            Z1 = df.pivot('y', 'x', 'm').values
            mask = df.pivot('y', 'x', 'mask')
            #Z1 *= 1 - mask
            xmin, xmax, ymin, ymax = df.x.min(), df.x.max(), df.y.min(), df.y.max()
            extent = xmax, xmin, ymax, ymin
            fig = plt.figure(frameon=False)
            im1 = plt.imshow(Z1, cmap=plt.cm.magma, alpha=.9, interpolation='nearest',
                             extent=extent, vmin=0, vmax=15, origin='lower')
            Z2 = openpiv.tools.imread(im)
            im2 = plt.imshow(Z2, cmap=plt.cm.gray, alpha=.6, interpolation='bilinear',
                             extent=extent)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.tight_layout(True)
            plt.axis('off')
            fig.savefig(output_path + im.split('/')[-1].split('.')[0] + "_heatmap_rel_with_beads.png",
                        bbox_inches='tight')
            fig.clear()

    def get_cellims_displacements_to_still_frame(self, cells_directory, beads_directory):
        cell_im_filelist = []
        all_displacements_list = []
        for folder in beads_directory:
            displacements_filelist = glob.glob(folder + '/*to_still_frame/*.csv')
            displacements_filelist.sort()
            all_displacements_list.append(displacements_filelist)
        for image in cells_directory:
            folder_cell_list = glob.glob(image + '/*.tif')
            folder_cell_list.sort()
            cell_im_filelist.append(folder_cell_list)
        return cell_im_filelist, all_displacements_list

    def get_beadsims_displacements_to_still_frame(self, beads_directory):
        bead_im_filelist = []
        all_displacements_list = []
        for folder in beads_directory:
            displacements_filelist = glob.glob(folder + '/*to_still_frame/*.csv')
            displacements_filelist.sort()
            all_displacements_list.append(displacements_filelist)
        for image in beads_directory:
            folder_cell_list = glob.glob(image + '/*.tif')
            folder_cell_list.sort()
            bead_im_filelist.append(folder_cell_list)
        return bead_im_filelist, all_displacements_list

    def make_heatmap_cell_overlays_to_still_frame(self, csv_im_pair_list):
        column_names = ['x', 'y', 'u', 'v', 'mask']
        op_tmp = '{0}_{1}_{2}_heatmap_to_still_cellz/'.format(str(self.window_size), str(self.overlap_size), str(self.search_area))
        for im, csv in csv_im_pair_list:
            output_path = os.path.join(self.output_path, op_tmp)
            #output_path = os.path.dirname(im) + os.sep + 'heatmap_to_still_cellz' + os.sep
            df = pd.read_csv(csv, '\t', header=None, names=column_names)
            df['m'] = np.sqrt(df['u'] ** 2 + df['v'] ** 2)
            Z1 = df.pivot('y', 'x', 'm').values
            mask = df.pivot('y', 'x', 'mask')
            mask_size = mask.values.sum()
            #Z1 *= 1 - mask
            xmin, xmax, ymin, ymax = df.x.min(), df.x.max(), df.y.min(), df.y.max()
            extent = xmax, xmin, ymax, ymin
            fig = plt.figure(frameon=False)
            im1 = plt.imshow(Z1, cmap=plt.cm.magma, alpha=.9, interpolation='nearest',
                             extent=extent, vmin=0, vmax=15, origin='lower')
            Z2 = openpiv.tools.imread(im)
            im2 = plt.imshow(Z2, cmap=plt.cm.gray, alpha=.3, interpolation='bilinear',
                             extent=extent)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.tight_layout(True)
            plt.axis('off')
            ofile = output_path + im.split('/')[-1].split('.')[0] + "_ms{0}".format(mask_size) \
                    + "_heatmap_with_cells_to_still.png"
            fig.savefig(ofile, bbox_inches='tight')
            fig.clear()

    def make_heatmap_bead_overlays_to_still_frame(self, csv_im_pair_list):
        column_names = ['x', 'y', 'u', 'v', 'mask']
        op_tmp = '{0}_{1}_{2}_heatmap_to_still_beads/'.format(str(self.window_size), str(self.overlap_size), str(self.search_area))
        for im, csv in csv_im_pair_list:
            output_path = os.path.join(self.output_path, op_tmp) #os.path.dirname(im) + os.sep + 'heatmap_to_still_beadz' + os.sep
            df = pd.read_csv(csv, '\t', header=None, names=column_names)
            df['m'] = np.sqrt(df['u'] ** 2 + df['v'] ** 2)
            Z1 = df.pivot('y', 'x', 'm').values
            mask = df.pivot('y', 'x', 'mask')
            mask_size = mask.values.sum()
            #Z1 *= 1 - mask

            xmin, xmax, ymin, ymax = df.x.min(), df.x.max(), df.y.min(), df.y.max()
            extent = xmax, xmin, ymax, ymin
            fig = plt.figure(frameon=False)
            im1 = plt.imshow(Z1, cmap=plt.cm.magma, alpha=.9, interpolation='nearest',
                             extent=extent, vmin=0, vmax=15, origin='lower')
            Z2 = openpiv.tools.imread(im)
            im2 = plt.imshow(Z2, cmap=plt.cm.gray, alpha=.3, interpolation='bilinear',
                             extent=extent)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.tight_layout(True)
            plt.axis('off')
            ofile = output_path + im.split('/')[-1].split('.')[0] + "_ms{0}".format(mask_size) \
                    + "_heatmap_with_beads_to_still.png"
            fig.savefig(ofile, bbox_inches='tight')
            fig.clear()

    def run(self, target_subfolder_list):
        # Make heatmaps over cells and beads, relative to still frame and between frames.
        # Need: list of displacements between frames and relative to still frame
        # List of cell images and bead images
        try:
            target_subfolder_list = list(map((lambda x: os.path.join(self.path, x)), target_subfolder_list))
            print(target_subfolder_list)
            for item in target_subfolder_list:
                beads_dirs = self.get_beads_dirs(item)
                cells_dirs = self.get_cells_difs(item)
                beads_dirs.sort(), cells_dirs.sort()
                self.calculate_displacements_for_folder_still_frame(beads_dirs)
                rel_cell_ims, rel_disp = self.get_cellims_displacements(cells_dirs, beads_dirs)
                rel_beads_ims, rel_disp = self.get_beadsims_displacements(beads_dirs)
                still_cell_im, still_disp = self.get_cellims_displacements_to_still_frame(cells_dirs, beads_dirs)
                still_beads_im, still_disp = self.get_beadsims_displacements_to_still_frame(beads_dirs)
                rel_cell = self.zip_up_ims_and_displacements(rel_cell_ims, rel_disp)
                rel_beads = self.zip_up_ims_and_displacements(rel_beads_ims, rel_disp)
                still_cell = self.zip_up_ims_and_displacements(still_cell_im, still_disp)
                still_beads = self.zip_up_ims_and_displacements(still_beads_im, still_disp)
                self.make_heatmap_cell_overlays(rel_cell)
                self.make_heatmap_bead_overlays(rel_beads)
                self.make_heatmap_cell_overlays_to_still_frame(still_cell)
                self.make_heatmap_bead_overlays_to_still_frame(still_beads)
        except (RuntimeError, ZeroDivisionError, ValueError, TypeError) as e:
            print(f"Exception: {str(e)}:\n" +
                f"{self.window_size}, {self.overlap_size}, {self.search_area} values not valid.")

if __name__ == "__main__":

    wnd = sys.argv[1] if(len(sys.argv) >= 2) else 8
    olp = sys.argv[2] if(len(sys.argv) >= 3) else 4
    scha = sys.argv[3] if(len(sys.argv) >= 4) else 8
    plt.ioff()



    # class ProcessInput(object):
    #     def __init__(self, b):
    #         self.b = b
    #
    #     def processInput(self, c):
    #         pivBatcher = PIVbatcher(
    #             '/AssayDev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
    #             c, self.b, c, '/Projects/PIV/res3'
    #         )
    #         pivBatcher.run(['test_sequence2'])
    #
    #
    # krange = [4, 5, 6, 7, 8]
    # for j in [2, 3, 4]:
    #     for k in krange:
    #         pi = ProcessInput(j)
    #         result = pi.processInput(k)

    # pivBatcher = PIVbatcher(
    #     '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
    #     8, 4, 8, '/allen/aics/modeling/jamies/projects/PIV/js_res09')
    # pivBatcher.run(['js_test09'])
    #
    # pivBatcher = PIVbatcher(
    #     '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
    #     8, 4, 8, '/allen/aics/modeling/jamies/projects/PIV/js_res10')
    # pivBatcher.run(['js_test10'])

    pivBatcher = PIVbatcher(
        '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180912_PDMS_data_to_analyze',
        16, 8, 16, '/allen/aics/modeling/jamies/projects/PIV/PDMS_01')
    pivBatcher.run(['Set1', 'Set2', 'Set3'])

    pivBatcher = PIVbatcher(
        '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180912_PDMS_data_to_analyze',
        24, 8, 24, '/allen/aics/modeling/jamies/projects/PIV/PDMS_02')
    pivBatcher.run(['Set1', 'Set2', 'Set3'])

    exit()
    pivBatcher = PIVbatcher(
        '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
        8, 4, 8, '/allen/aics/modeling/jamies/projects/PIV/js_res12')
    pivBatcher.run(['js_test12'])

    pivBatcher = PIVbatcher(
        '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
        8, 4, 8, '/allen/aics/modeling/jamies/projects/PIV/js_res14')
    pivBatcher.run(['js_test14'])

    pivBatcher = PIVbatcher(
        '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
        8, 4, 8, '/allen/aics/modeling/jamies/projects/PIV/js_res15')
    pivBatcher.run(['js_test15'])

    pivBatcher = PIVbatcher(
        '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
        8, 4, 8, '/allen/aics/modeling/jamies/projects/PIV/js_res17')
    pivBatcher.run(['js_test17'])

    pivBatcher = PIVbatcher(
        '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
        8, 4, 8, '/allen/aics/modeling/jamies/projects/PIV/js_res18')
    pivBatcher.run(['js_test18'])

    pivBatcher = PIVbatcher(
        '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
        8, 4, 8, '/allen/aics/modeling/jamies/projects/PIV/js_res19')
    pivBatcher.run(['js_test19'])

    pivBatcher = PIVbatcher(
        '/allen/aics/assay-dev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
        8, 4, 8, '/allen/aics/modeling/jamies/projects/PIV/js_res20')
    pivBatcher.run(['js_test20'])

    #
    # exit()

    # pivBatcher = PIVbatcher(
    #     '/AssayDev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2',
    #     wnd, olp, scha, '/Projects/PIV/res'
    # )
    # pivBatcher.run(['test_sequence2'])  #'20171128/all_crops/C1-20171128_M02_003_19_cell1_beads'])  # ['test_sequence'])

    # path_base = '/AssayDev/MicroscopyOtherData/Melissa/20180116_PIV_slides/20171220_PIV_data_round2'
    # day1_MG_all = os.path.join(path_base, '20171121/MG/all_crops')
    # day1_PDL_lam_all = os.path.join(path_base, '20171121/PDL_lam/all_crops')
    # day2_MG_all = os.path.join(path_base, '20171128/all_crops')
    # test_dir = os.path.join(path_base, 'test_sequence')  # Testing
