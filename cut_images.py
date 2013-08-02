# coding=utf-8
__author__ = 'Roman Podlinov'

import os
import fnmatch
import numpy as np

from scipy.misc import imread
from skimage.io import imsave
from skimage.color import rgb2gray
from numpy.lib.stride_tricks import as_strided


def rolling_window_2d(A, xsize = 3, ysize = 3):
    """
    rolling window for 2D arrays. doesn't work properly if step >1
    """
    xstep = ystep = 1
    return as_strided(A, ((A.shape[0] - xsize + 1) / xstep, (A.shape[1] - ysize + 1) / ystep, xsize, ysize),
                             (A.strides[0] * xstep, A.strides[1] * ystep, A.strides[0], A.strides[1]))


def read_images_in_dir(dir, load_pattern = '.png:.jpg:.jpeg:.gif'):
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


def save_sub_images(image, filename = 'image_%04d.jpg', h_size = 3, v_size = 3):
    i = 0
    for r in range(0, image.shape[0], v_size):
        for c in range(0, image.shape[1], h_size):
            if (c + h_size ) > image.shape[1]:
                if (r + v_size ) > image.shape[0]:
                    imsave(filename % i, image[image.shape[0] - v_size:, image.shape[1] - h_size:])
                else:
                    imsave(filename % i, image[r:r + v_size, image.shape[1] - h_size:])
            else:
                if (r + v_size ) > image.shape[0]:
                    imsave(filename % i, image[image.shape[0] - v_size:, c:c + h_size])
                else:
                    imsave(filename % i, image[r:r + v_size, c:c + h_size])
            i +=1


def main():
    in_dir = './in/'
    out_dir = './out/'

    # A = np.arange(120).reshape((10, 12))
    # print A
    # save_sub_images(A, out_dir+'image_%04d.jpg')

    image_files = read_images_in_dir(in_dir)
    for filename in image_files:
        print filename.split('.')[0] + '_%04d.jpg'
        # im = imread(os.path.join(in_dir, filename), as_grey= True)
        im = imread(os.path.join(in_dir, filename))
        im = rgb2gray(im)
        save_sub_images(im, out_dir + filename.split('.')[0] + '_%04d.jpg', 30, 30)



if __name__ == "__main__":
    main()