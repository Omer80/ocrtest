# coding=utf-8
__author__ = 'Roman Podlinov'

import os
import fnmatch
import numpy as np

from scipy.misc import imread
from skimage.io import imsave
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

def main():
    in_dir = './in/'
    out_dir = './out/'

    # image_files = read_images_in_dir(in_dir)
    # print image_files
    #
    # for filename in image_files:
    #     im = imread(os.path.join(in_dir, filename), as_grey= True)
    #     all_windows = rolling_window_2d(im, 30, 30)
    #     n = 0
    #     for window in all_windows
    #
    #     imsave(os.path.join(out_dir,'image%06d.jpg' % i), im)
    #     n += 1

    A = np.arange(100).reshape((10, 10))
    print A
    A.reshape((3,3))
    print A


if __name__ == "__main__":
    main()