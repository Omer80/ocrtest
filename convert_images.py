# coding=utf-8
__author__ = 'Roman Podlinov'
import numpy as np
import os

from skimage.io import ImageCollection, imread, imsave
from skimage.color import rgb2gray
from skimage.transform import resize

"""
Reads images from IN directory, converts to gray, resizes and stores as jpg into OUT directory

"""

def img_convert(f):
    return resize(rgb2gray(imread(f)),(30,30))

in_dir = './in/'
out_dir = './out/'

icol = ImageCollection(in_dir+'*.jpg:'+in_dir+'*.png:'+in_dir+'*.gif', load_func=img_convert)
i = 0
for im in icol:
    imsave(os.path.join(out_dir,'image%06d.jpg' % i), im)
    i += 1

print len(icol)
