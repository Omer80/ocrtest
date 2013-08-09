# coding=utf-8
__author__ = 'Roman Podlinov'

import os
import fnmatch
import numpy as np

from scipy.misc import imread
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage.transform import resize



class ImagesPreparationTool:


    def __read_images_in_dir(self, dir, load_pattern = '.png:.jpg:.jpeg:.gif'):
        load_pattern = load_pattern.split(':')
        matches = []
        for root, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                # if filename.endswith(('.jpg', '.jpeg', '.gif', '.png')):
                if filename.endswith(tuple(load_pattern)):
                    matches.append(filename)

        # return sorted(matches, key=lambda item: (int(item.partition(' ')[0])
        #                            if item[0].isdigit() else float('inf'), item))

        return matches


    def __slice_image_to_small_images(self, image, filename = 'image_%04d.jpg', h_size = 3, v_size = 3):
        i = 0
        for r in range(0, image.shape[0], v_size):
            for c in range(0, image.shape[1], h_size):
                if (c + h_size ) > image.shape[1]:
                    if (r + v_size ) > image.shape[0]:
                        im = image[image.shape[0] - v_size:, image.shape[1] - h_size:]
                    else:
                        im = image[r:r + v_size, image.shape[1] - h_size:]
                else:
                    if (r + v_size ) > image.shape[0]:
                        im = image[image.shape[0] - v_size:, c:c + h_size]
                    else:
                        im = image[r:r + v_size, c:c + h_size]

                if sum(im).all() > 0:
                    imsave(filename % i, im)
                    i +=1

    def slice_images(self, dir_in, dir_out, h_size, v_size):
        """
        1. Read images from directory and slice they into small images
        """
        # test
        # A = np.arange(120).reshape((10, 12))
        # print A
        # slice_image_to_small_images(A, out_dir+'image_%04d.jpg')
        image_files = self.__read_images_in_dir(dir_in)
        for filename in image_files:
            print filename.split('.')[0] + '_%04d.jpg'
            # im = imread(os.path.join(in_dir, filename), as_grey= True)
            im = imread(os.path.join(dir_in, filename))
            im = rgb2gray(im)
            self.__slice_image_to_small_images(im, dir_out + filename.split('.')[0] + '_%04d.jpg', h_size, v_size)

    def resize_images(self, dir_in, dir_out, h_size, v_size, make_grey = True):
        image_files = self.__read_images_in_dir(dir_in)
        i = 0
        for filename in image_files:
            print filename.split('.')[0] + '_%04d.jpg'
            # im = imread(os.path.join(in_dir, filename), as_grey= True)
            im = resize(imread(os.path.join(dir_in, filename)), (h_size,v_size))
            if make_grey:
                im = rgb2gray(im)
            imsave(os.path.join(dir_out, 'image%05d.jpg' % i), im)






def main():
    # in_dir = './in/'
    out_dir = './out/'

    tool = ImagesPreparationTool()
    # tool.slice_images("/mnt/hgfs/Virtual Machines/selected_images/", out_dir, 18, 18)
    # tool.resize_images("/mnt/hgfs/Virtual Machines/twitter_logos_orig/", out_dir, 18, 18)

    im = np.arange(120).reshape((10, 12))
    print im
    im = im.reshape()




if __name__ == "__main__":
    main()