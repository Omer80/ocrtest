# coding=utf-8
import os
import sys
import cv2
import logging
import numpy

from misc.file_helper import FileHelper


class SIFTComparer:
    MIN_MATCH_COUNT = 3
    __templatesFeatures = None

    def __init__(self, templates_dir):
        self.templates_dir = templates_dir
        if templates_dir is None or not os.path.exists(templates_dir):
            logging.critical("Invalid path: {0}".format(templates_dir))
            sys.exit(1)

        # self.detector = cv2.SIFT(3200)
        self.detector = cv2.SIFT(900, edgeThreshold=30)
        # matcher = cv2.BFMatcher(cv2.NORM_L2)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        if self.__templatesFeatures is None:
            self.__templatesFeatures =  self.__getFeaturesFromTemplates()

    def __getFeaturesFromTemplates(self):

        features = {}
        images = FileHelper.read_images_in_dir(self.templates_dir)
        for filename in images:
            im = cv2.imread(os.path.join(self.templates_dir, filename), cv2.IMREAD_GRAYSCALE)
            kp, desc = self.detector.detectAndCompute(im, None)
            logging.debug('{0} - {1} features'.format(filename, len(kp)))
            # features.append((kp, desc))
            features[filename] = (kp, desc)
        return features

    def compareAgainstTmplts(self, image):
        result = {}
        if not self.__templatesFeatures:
            logging.critical("No templates to compare")
            sys.exit(1)

        kp2, desc2 = self.detector.detectAndCompute(image, None)
        logging.debug('img1 - %d features' % len(kp2))
        for name, (kp1, desc1) in self.__templatesFeatures.iteritems():
            raw_matches = self.matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
            kp_pairs = self.__filter_matches(kp1, kp2, raw_matches)
            if len(kp_pairs) >= self.MIN_MATCH_COUNT:
                result[name] = len(kp_pairs)
        return result

    def __filter_matches(self, kp1, kp2, matches, ratio=0.75):
        """Filters features that are common to both images"""
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append(kp1[m.queryIdx])
                mkp2.append(kp2[m.trainIdx])
        kp_pairs = zip(mkp1, mkp2)
        return kp_pairs

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    obj = SIFTComparer('../logos/images')
    img1 = cv2.imread('../logos/images/image000000.jpg', cv2.IMREAD_GRAYSCALE)
    print obj.compareAgainstTmplts(img1)
