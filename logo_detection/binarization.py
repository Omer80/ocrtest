import os
import cv2
import itertools

from misc.file_helper import FileHelper


def load_image_adaptive(filename, neighbours=5, shift=8, inverse=False, binary_type=None):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if binary_type is None:
        if inverse:
            binary_type = cv2.THRESH_BINARY_INV
        else:
            binary_type = cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, binary_type, neighbours, shift)
    # return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, neighbours, shift)


def load_image_thresholding(filename, threshold=127, binary_type=cv2.THRESH_BINARY):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    thresholdValue, image = cv2.threshold(image, threshold, 255, binary_type)
    return image


def binarize_folder_adaptive(inFolder, outFolder):
    fileList = FileHelper.read_images_in_dir(inFolder)
    FileHelper.create_or_clear_dir(outFolder)
    for filename in fileList:
        image = load_image_adaptive(os.path.join(inFolder, filename))
        cv2.imwrite(os.path.join(outFolder, filename), image)


def binarize_folder_threshold(inFolder, outFolder):
    fileList = FileHelper.read_images_in_dir(inFolder)
    FileHelper.create_or_clear_dir(outFolder)
    for filename in fileList:
        image = load_image_thresholding(os.path.join(inFolder, filename), 190)
        cv2.imwrite(os.path.join(outFolder, filename), image)


def binarize_file_adaptive(filename, outFolder):
    FileHelper.create_or_clear_dir(outFolder)
    fullFilename = filename
    path, filename = os.path.split(filename)
    filename_base, filename_ext = os.path.splitext(filename)
    for neighbours, shift in itertools.product(range(3, 17, 2), range(10)):
        image = load_image_adaptive(fullFilename, neighbours, shift, binary_type=cv2.THRESH_TRUNC)
        ffsave = '%s_%d_%d%s' % (filename_base, neighbours, shift, filename_ext)
        cv2.imwrite(os.path.join(outFolder, ffsave), image)


def binarize_file_threshold(filename, outFolder):
    FileHelper.create_or_clear_dir(outFolder)
    fullFilename = filename
    path, filename = os.path.split(filename)
    filename_base, filename_ext = os.path.splitext(filename)
    for threshold in range(30, 200, 20):
        image = load_image_thresholding(fullFilename, threshold, binary_type=cv2.THRESH_BINARY)
        ffsave = '%s_%d%s' % (filename_base, threshold, filename_ext)
        cv2.imwrite(os.path.join(outFolder, ffsave), image)


if __name__ == '__main__':
    # binarize_folder_adaptive('/home/valeriy/projects/hashtag/logos/frames/', '/home/valeriy/projects/hashtag/logos/bin_frames')
    # binarize_file_adaptive('/home/valeriy/projects/hashtag/logos/frames/5_00450.jpg', '/home/valeriy/projects/hashtag/logos/bin_frames')

    binarize_folder_threshold('/home/valeriy/projects/hashtag/logos/frames/', '/home/valeriy/projects/hashtag/logos/bin_frames')
    # binarize_file_threshold('/home/valeriy/projects/hashtag/logos/frames/5_00450.jpg', '/home/valeriy/projects/hashtag/logos/bin_frames')
    # binarize_file_threshold('/home/valeriy/projects/hashtag/logos/frames/5_00735.jpg', '/home/valeriy/projects/hashtag/logos/bin_frames')
