# coding=utf-8
__author__ = 'Roman Podlinov'

import os

from skimage.io import ImageCollection, imread, imsave
from skimage.color import rgb2gray
from skimage.transform import resize

"""
Reads images from IN directory, converts to gray, resizes and stores as jpg into OUT directory

"""

def main():
    # in_dir = './in/'
    in_dir = '/mnt/hgfs/Virtual Machines/for_training/imageclipper'
    out_dir = './out/'
    load_pattern = '.png:.jpg:.jpeg:.gif'

    load_pattern = load_pattern.split(':')
    matches = []
    for root, dirnames, filenames in os.walk(in_dir):
        for filename in filenames:
            # if filename.endswith(('.jpg', '.jpeg', '.gif', '.png')):
            if filename.endswith(tuple(load_pattern)):
                parts = filename.split('.')
                print parts[0]+'#.'+parts[-1]
                os.rename(os.path.join(in_dir, filename), os.path.join(in_dir, parts[0]+'#.'+parts[-1]))



if __name__ == "__main__":
    main()