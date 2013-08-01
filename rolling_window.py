# coding=utf-8
__author__ = 'Roman Podlinov'

import matplotlib.pyplot as plt
import numpy as np
# import time
import logging
import scipy
import os

from skimage.io import imsave
from scipy.misc import imread
from skimage.color import rgb2gray
from skimage import filter
from numpy.lib.stride_tricks import as_strided

# def moving_average(Ic, filtsize):
#     Im = np.empty(Ic.shape, dtype='Float64')
#     # scipy.ndimage.filters.uniform_filter(Ic, filtsize, output=Im)
#     scipy.ndimage.filters.sobel(Ic, output=Im)
#     return Im

def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to np-discussion.
    <http://www.mail-archive.com/np-discussion@scipy.org/msg29450.html>"""
    if window < 1:
       raise ValueError, "`window` must be at least 1."
    if window > a.shape[-1]:
       raise ValueError, "`window` is too long."
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window_multi_d(a, window):
    """
    rolling window for multi dimentional arrays d>=2

    Example:
    filtsize = (30, 30)
    b = rolling_window_multi_d(im, filtsize)
    blurred = b.mean(axis=-1).mean(axis=-1)
    """
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a


def rolling_window_2d(A, xsize = 3, ysize = 3):
    """
    rolling window for 2D arrays. doesn't work properly if step >1
    """
    xstep = ystep = 1
    return as_strided(A, ((A.shape[0] - xsize + 1) / xstep, (A.shape[1] - ysize + 1) / ystep, xsize, ysize),
                             (A.strides[0] * xstep, A.strides[1] * ystep, A.strides[0], A.strides[1]))


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=10)

    src_im = imread('./test_img/784_00693.jpg')
    im = rgb2gray(src_im)
    print im.shape

    # A = np.arange(100).reshape((10, 10))
    # print A
    # all_windows = rolling_window_2d(A, 5, 5)
    all_windows = rolling_window_2d(im, 30, 30)
    # print all_windows
    print all_windows.shape
    n = 0
    for i in xrange(0,all_windows.shape[0],2):
        for j in xrange(0,all_windows.shape[1],2):

            if (sum(all_windows[i,j])).all() == 0: #skip empty black images
                continue
            if i == j: #store to disc only diagonal images
                imsave(os.path.join('./out/', 'image%06d.jpg' % n), all_windows[i,j])
            n += 1
    print n


if __name__ == "__main__":
    main()