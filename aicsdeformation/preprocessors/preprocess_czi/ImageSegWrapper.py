import cv2
import numpy as np
import aicsimageio
import pathlib as plib
import imageio as imio
from matplotlib import pyplot as plt
import os


class ImgSegWrapper(aicsimageio.AICSImage):
    def __init__(self, data, **kwargs):
        super(ImgSegWrapper, self).__init__(data, **kwargs)
        self.czi_name = data

    def time_range(self):
        t_index = self.__get_index_order('T')
        return self.shape[t_index]

    def channel_range(self):
        sc_index = self.__get_index_order('C')
        return self.shape[sc_index]

    def perform_alignment(self, channel: int = 0):
        pth = '../../Data/tmp/frm'
        os.mkdir('../../Data/tmp')
        slice_array = self.generate_bead_slices()
        frame_k_dict = self.cv_find_points(slice_array)
        pts_arr = []
        img0 = slice_array[0]
        self.write_slice(pth, 0, img0)
        for idx in range(1,len(frame_k_dict)):
            pts = self.align_pair(frame_k_dict[idx], frame_k_dict[0], slice_array[idx], slice_array[0])
            pts_arr.append(pts)
            w_img = cv2.warpPerspective(slice_array[idx], pts, (img0.shape[1], img0.shape[0]))
            self.write_slice(pth, idx, w_img)
        print("implement alignment here")

    def write_slice(self, path, idx, img):
        fpath = path + str(idx).zfill(3) + '.jpg'
        imio.imwrite(fpath, img)

    def align_pair(self, a: dict, b: dict, src_img, dst_img, draw:bool=False):
        MIN_MATCH_COUNT = 50
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(a['descriptors'], b['descriptors'], k=2)
        #
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            src_key_pts = a['key_points']
            dst_key_pts = b['key_points']
            src_pts = np.float32([src_key_pts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([dst_key_pts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            matchesMask = mask.ravel().tolist()

            h, w = src_img.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None

        if draw:
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=None,
                               matchesMask=matchesMask,
                               flags=2)
            img3 = cv2.drawMatches(src_img, src_key_pts, dst_img, dst_key_pts, good, None, **draw_params)
            plt.imshow(img3, 'gray'), plt.show()
        return M

    @staticmethod
    def cv_find_points(slice_arr: list):
        ans = []
        n_of_frames = len(slice_arr)
        sift = cv2.xfeatures2d.SIFT_create()
        try:
            for f_i in range(0, n_of_frames):
                img = cv2.normalize(slice_arr[f_i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                kp, des = sift.detectAndCompute(img, None)
                ans.append({'key_points': kp, 'descriptors': des})
        except Exception as e:
            print("Big Bad ERROR!!! ", e)
            exit()
        return ans

    def find_points(self, slice_arr: list):
        frame_points = []
        (frames, y_max, x_max) = slice_arr.shape
        for f_i in range(0, frames):
            point_list = []
            for y_i in range(0, y_max):
                for x_i in range(0, x_max):
                    if slice_arr[f_i, y_i, x_i] > 0.5: point_list.append([y_i, x_i])
            frame_points.append(point_list)
        return frame_points

    def generate_bead_slices(self, fname:str = None, c_i: int = 0):
        if fname is None: fname = self.czi_name
        ti_max = self.time_range()
        slice_array = []
        mp_array = []
        dcube = self.get_image_data(out_orientation="TZYX", C=c_i)
        for t_i in range(0, ti_max):
            z_idx = self.find_beed_slice(dcube[t_i, :, :, :])
            i_name = self.slice_name_from_file(fname, 'beads', t_i, c_i, z_idx)
            if t_i == 0: plib.Path.mkdir(i_name.parent, parents=True, exist_ok=True)
            slice_array.append(self.create_image_slice(i_name, t_i, c_i, z_idx, dcube))
            mp_array.append(np.max(dcube[t_i, :, :, :], axis=0))
        return slice_array

    def create_image_slice(self, fpath, t_idx: int=0, c_idx: int=0, z_idx: int=0, dcube=None):
        if dcube is None:
            l_data = self.get_image_data(out_orientation="ZYX", T=t_idx, C=c_idx)
        else:
            l_data = dcube[t_idx, :, :, :]  # dcube is ordered TZYX
        if z_idx < 0:
            return None
        slice_d = l_data[z_idx, :, :]
        imio.imwrite(fpath, slice_d)
        return slice_d

    def find_beed_slice(self, tmp_data):
        z_data = np.sum(tmp_data, axis=(1, 2))
        z_data = z_data/np.max(z_data)
        dz2 = np.gradient(np.gradient(z_data))
        z_ind = np.argmax(z_data)
        dz2_ind = np.argmin(dz2)
        if z_ind == dz2_ind:
            idx = z_ind
        elif dz2[dz2_ind] < 0.0:
            idx = z_ind
        else:
            idx = -1
        return idx


    def slice_name_from_file(self, fname:str, label: str,
                             time_idx: int, scene_idx: int, z_idx: int):
        path = plib.Path(fname).parent
        folder_name = plib.Path(path, plib.PurePosixPath(fname).stem)
        scene = 'Scene_' + str(scene_idx).zfill(4)
        filen = label + '_' + str(time_idx).zfill(4) + '_z' + str(z_idx).zfill(3) + '.jpg'
        ofname = folder_name.joinpath(scene, filen)
        return ofname

    def __get_index_order(self, letter):
        try:
            t_index = self.dims.index(letter)
        except ValueError:
            print('Error: image file needs to contain a time axis to do temporal alignment!')
            exit(0)
        return t_index
