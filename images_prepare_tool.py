# coding=utf-8
__author__ = 'Roman Podlinov'

import os
import fnmatch
import numpy as np

from scipy.misc import imread
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage.transform import resize
from process_image import Image
from file_helper import FileHelper


class ImagesPreparationTool:

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
        image_files = FileHelper.read_images_in_dir(dir_in)
        i = 0
        for filename in image_files:
            print filename.split('.')[0] + '_%04d.jpg'
            # im = imread(os.path.join(dir_in, filename))
            # im = rgb2gray(im)
            im = Image(os.path.join(dir_in, filename))
            im.prepare()

            self.__slice_image_to_small_images(im.image, os.path.join(dir_out, filename.split('.')[0] + '_%04d.jpg'), h_size, v_size)

    def resize_images(self, dir_in, dir_out, h_size, v_size, make_grey = True):
        image_files = FileHelper.read_images_in_dir(dir_in)
        i = 0
        for filename in image_files:
            print filename.split('.')[0] + '_%04d.jpg'
            # im = imread(os.path.join(in_dir, filename), as_grey= True)
            im = resize(imread(os.path.join(dir_in, filename)), (h_size,v_size))
            if make_grey:
                im = rgb2gray(im)
            imsave(os.path.join(dir_out, 'image%05d.jpg' % i), im)
            i += 1

    def __image_to_csv_string(self, im, classification_value):
        return ','.join("{0}".format(item) for item in im.ravel())+','+str(classification_value)

    def images_to_csv_file(self, dir_positive, dir_negative, filename_positive = 'positive.csv', filename_negative = 'negative.csv'):
        positive_images = FileHelper.read_images_in_dir(dir_positive)
        negative_images = FileHelper.read_images_in_dir(dir_negative)
        if (len(negative_images) > len(positive_images) * 3 ):
            negative_images = negative_images[:len(positive_images)* 3]
            # print "cut negative images"

        with open(filename_positive,"wb") as f:
            for filename in positive_images:
                im = imread(os.path.join(dir_positive, filename), True)
                f.write(self.__image_to_csv_string(im,1))
                f.write('\n')

        with open(filename_negative, "w+b") as f:
            for filename in negative_images:
                im = imread(os.path.join(dir_negative, filename), True)
                f.write(self.__image_to_csv_string(im,0))
                f.write('\n')


def main():
    # in_dir = './in/'
    out_dir = './out/'

    tool = ImagesPreparationTool()

    # hashtags preparation
    # tool.slice_images("/mnt/hgfs/Virtual Machines/selected_images/", out_dir, 18, 18)
    # tool.resize_images("/mnt/hgfs/Virtual Machines/twitter_logos_orig/", out_dir, 18, 18)
    # tool.images_to_csv_file('/mnt/hgfs/Virtual Machines/for_training/positive/1', '/mnt/hgfs/Virtual Machines/for_training/negative/', './out/training.csv')

    # twitter logos preparation
    # tool.resize_images("/mnt/hgfs/Virtual Machines/twitter_logos/", "/mnt/hgfs/Virtual Machines/for_training/positive/4", 30, 30)
    # tool.slice_images("/mnt/hgfs/Virtual Machines/selected_images/", "/mnt/hgfs/Virtual Machines/for_training/negative2", 30, 30)
    tool.images_to_csv_file('/mnt/hgfs/Virtual Machines/for_training/positive/4', '/mnt/hgfs/Virtual Machines/for_training/negative2/', './out/logo_positive.csv', './out/logo_negative.csv')


if __name__ == "__main__":
    main()