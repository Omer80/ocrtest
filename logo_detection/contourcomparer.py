# coding=utf-8
from misc.file_helper import FileHelper
from collections import namedtuple

import os
import cv2
import binarization
import logging
import itertools
import numpy as np

ImageContours = namedtuple('ImageContours', ['contours', 'hierarchy'])

class Contour(object):
    def __init__(self, contour):
        self.contour = contour
        self.solidity = Contour.solidityCalculation(contour)
        self.extent = Contour.extentCalculation(contour)

    @staticmethod
    def solidityCalculation(contour):
        """ Solidity is the ratio of contour area to its convex hull area."""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0.0:
            return 0.0
        return float(area) / hull_area

    @staticmethod
    def extentCalculation(contour):
        """Extent is the ratio of contour area to bounding rectangle area."""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area == 0.0:
            return 0.0
        return float(area) / rect_area

    @staticmethod
    def convert(contoursList):
        result = []
        for c in contoursList:
            result.append(Contour(c))
        return result



class ContourComparer:

    def __init__(self, templates_dir):
        self.templates_dir = templates_dir
        self.contour_mode = cv2.RETR_TREE
        self.template_contour_mode = cv2.RETR_EXTERNAL
        self.contour_method = cv2.CHAIN_APPROX_SIMPLE
        self.templates = self.__load_prepare_templates()

    def __load_prepare_templates(self):
        """Load templates. Template MUST be white on black background"""
        templates = []
        filelist = sorted(FileHelper.read_images_in_dir(self.templates_dir))
        for filename in filelist:
            templ = self.__load_template(os.path.join(self.templates_dir, filename))
            if templ is not None:
                templates.append(templ)

        return templates

    def __load_template(self, filename):
        image = self.load_image(filename)
        template_contours, template_hierarchy = cv2.findContours(image, self.template_contour_mode,
                                                                    self.contour_method)
        if len(template_contours) > 1:
            logging.warning('Template is too complex %s' % (filename,))
            return None

        # template_contours = template_contours[0]
        # print '%0.3f %3d %0.2f %0.2f ' % (0.0, len(template_contours), contourSolidity(template_contours), contourExtent(template_contours))

        return ImageContours(Contour.convert(template_contours), template_hierarchy)


    def load_image(self, filename):
        # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # (thresh, img_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # return img_bw
        return binarization.load_image_thresholding(filename, 127)

    def compareAgainstTmplts(self, image, debugFilename=None):
        for threshold, inverse in itertools.product(range(20, 201, 20), (False, True)):
            bimage = ContourComparer.binaryze(image, threshold, inverse)
            contours, hierarchy = cv2.findContours(bimage.copy(), self.contour_mode, self.contour_method)

            imagec = ImageContours(Contour.convert(contours), hierarchy)
            for template in self.templates:
                matches = self._compare_image_with_template(template, imagec)
                if matches is not None and len(matches) > 0:
                    #todo: try not only first, but all with matching score less then 0.5
                    matchCoef = matches[0][0]
                    matchContour = matches[0][1]
                    if ContourComparer.compareDecision(template.contours[0], matchContour, matchCoef):
                        logging.debug("match_coef:{0} len:{1} solidity:{2} extent:{3}".format(matchCoef, len(matchContour.contour), matchContour.solidity, matchContour.extent))
                        if debugFilename is not None:
                            bimage = ContourComparer.drawContours(bimage, [matchContour.contour])
                            cv2.imwrite(debugFilename, bimage)
                        return True

        return False

    def _compare_image_with_template(self, template, image, debugImage=None, debugImageFilename=None):
        if len(template.contours) != 1:
            logging.warning('Try of using incorrect template. Template MUST contain exactly one contour')
            return -1

        result = []
        templateContour = template.contours[0]
        for c in image.contours:
            matchingCoef = ContourComparer._match_two_contours(templateContour.contour, c.contour)
            if matchingCoef > 0.0:
                result.append((matchingCoef, c))

        result = sorted(result, key=lambda x: x[0])

        if debugImage is not None and debugImageFilename is not None:
            #todo: draw first three best candidates
            pass

        return result

    @staticmethod
    def compareDecision(template, contour, matchingCoef):
        # print '%0.3f %3d %0.2f %0.2f ' % (matchingCoef, len(template.contour), template.solidity, template.extent)
        # print '%0.3f %3d %0.2f %0.2f ' % (matchingCoef, len(contour.contour), contour.solidity, contour.extent)
        if matchingCoef > 0.04:
            return False
        if abs(template.solidity - contour.solidity) > 0.04 or abs(template.extent - contour.extent) > 0.04:
            return False
        if len(contour.contour) > len(template.contour)*5:
            return False
        return True

    @staticmethod
    def binaryze(image, threshold, inverse=False):
        if inverse:
            binary_type = cv2.THRESH_BINARY_INV
        else:
            binary_type = cv2.THRESH_BINARY
        thresholdValue, bimage = cv2.threshold(image, threshold, 255, binary_type)
        return bimage

    @staticmethod
    def _match_two_contours(rawContour1, rawContour2):
        return cv2.matchShapes(rawContour1, rawContour2, 1, 1.0)

    @staticmethod
    def drawContours(image, contours):
        # print len(contours)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

        return image

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    obj = ContourComparer('../logos/bw')
    img1 = cv2.imread('../logos/images/image000000.jpg', cv2.IMREAD_GRAYSCALE)
    print obj.compareAgainstTmplts(img1)

    # template_dir = '../logos/bw'
    # test_dir = '/home/valeriy/projects/hashtag/logos/test'
    # # template_dir = '../logos'
    # # test_dir = '../in'
    #
    # # filename = '/home/valeriy/projects/hashtag/logos/twitter_frames/784_00492.jpg'
    # # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # FileHelper.create_or_clear_dir('/home/valeriy/projects/hashtag/logos/bin_twitter_frames/')
    #
    # comparer = ContourComparer(template_dir)
    # checkDir = '/home/valeriy/projects/hashtag/logos/twitter_frames/'
    # outDir = '/home/valeriy/projects/hashtag/logos/bin_twitter_frames/'
    # for filename in sorted(FileHelper.read_images_in_dir(checkDir)):
    #     image = cv2.imread(os.path.join(checkDir, filename), cv2.IMREAD_GRAYSCALE)
    #     if comparer.compareImageWithTemplates(image, os.path.join(outDir, filename)):
    #         print 'Yes: ' + filename
    #     else:
    #         print 'No: ' + filename