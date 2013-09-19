import os
import logging

import cv2
import itertools

import ocr_utils
from file_helper import FileHelper


class ContourComparer(object):
    def __init__(self, templates_dir, images_dir, results_dir='./results', contour_mode=cv2.RETR_EXTERNAL, contour_method=cv2.CHAIN_APPROX_NONE):
        self.templates_dir = templates_dir
        self.images_dir = images_dir
        self.results_dir = results_dir

        self.contour_mode = contour_mode
        self.contour_method = contour_method

        if os.path.exists(results_dir):
            FileHelper.remove_files_in_dir(results_dir)
        else:
            os.makedirs(results_dir)

        self._load_prepare_templates()

    def _load_prepare_templates(self):
        self.contours = []
        for filename in sorted(FileHelper.read_images_in_dir(self.templates_dir)):
            image = cv2.imread(os.path.join(self.templates_dir, filename), cv2.IMREAD_GRAYSCALE)
            contours, hierarchy = cv2.findContours(image, self.contour_mode, self.contour_method)
            self.contours.append((filename, contours, hierarchy))

    def compareImageWithTemplates(self, image):
            image_contours, image_hierarchy = cv2.findContours(image, self.contour_mode, self.contour_method)
            result = []
            for name, template_contours, template_hierarchy in self.contours:
                matchingCoef = cv2.matchShapes(image_contours[0], template_contours[0], 1, 1.0)
                logging.debug('Image matching coefficient with %s is %f' % (name, matchingCoef))
                result.append((matchingCoef, name))
            return min(result)


if __name__ == '__main__':
    ocr_utils.init_console_logging(logging.INFO)
    # template_dir = '/home/valeriy/projects/hashtag/logos/learn'
    # test_dir = '/home/valeriy/projects/hashtag/logos/test'
    template_dir = '../logos'
    test_dir = '../in'

    # comparer = ContourComparer(template_dir, test_dir)

    # for filename in FileHelper.read_images_in_dir(test_dir):
    #     image = cv2.imread(os.path.join(test_dir, filename), cv2.IMREAD_GRAYSCALE)
    #     logging.info('Found match for %s: %s' % (filename, str(comparer.compareImageWithTemplates(image))))

    image = cv2.imread(os.path.join(test_dir, 'twitter_logo1.jpg'), cv2.IMREAD_GRAYSCALE)
    contours_find_modes = [cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE]
    contours_find_methods = [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_TC89_KCOS]
    for mode, method in itertools.product(contours_find_modes, contours_find_methods):
        comparer = ContourComparer(template_dir, test_dir, contour_mode=mode, contour_method=method)
        match = comparer.compareImageWithTemplates(image.copy())
        logging.info('Math with params (%d, %d) is %s' % (mode, method, str(match)))
