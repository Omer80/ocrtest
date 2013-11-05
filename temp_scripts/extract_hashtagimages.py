import os
from skimage.io import imread, imsave

from misc.file_helper import FileHelper
from dataset.image_folder import getTagCoordinates


def hashtagGetGenerator(folder):
    for dir, filename in FileHelper.read_images_in_dir_recursively(folder):
        if dir[-1] == '\\':
            dir = dir[:-1]
        path, dwt = os.path.split(dir)

        yield dir, filename, getTagCoordinates(dwt)


if __name__ == '__main__':
    import sys
    for dir, filename, tag in hashtagGetGenerator(sys.argv[1]):
        # print filename, tag
        tag = [tag[0], tag[1], tag[2], tag[1] + (tag[2] - tag[0])]
        image = imread(os.path.join(dir, filename))

        ht_image = image[tag[0]:tag[2], tag[1]:tag[3], :]

        path, dwt = os.path.split(dir)
        fbasename, ext = os.path.splitext(filename)

        try:
            os.makedirs(os.path.join(sys.argv[2], dwt))
        except OSError:
            pass
        imsave(os.path.join(sys.argv[2], dwt, fbasename+'_ht'+ext), ht_image)
        # imsave(os.path.join(sys.argv[2], dwt, fbasename+'_ht'+ext), ht_image)
