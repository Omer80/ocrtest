# coding=utf-8
__author__ = 'Roman Podlinov'

import os
import sys
import cv2
import logging
import numpy

from file_helper import FileHelper


class SIFTComparer:
    MIN_MATCH_COUNT = 3

    def __init__(self, templates_dir, images_dir, results_dir = './results'):
        self.templates_dir = templates_dir
        self.images_dir = images_dir
        self.results_dir = results_dir
        if templates_dir is None or not os.path.exists(templates_dir):
            logging.error("Invalid path: {0}".format(templates_dir))
            sys.exit(1)

        if images_dir is None or not os.path.exists(images_dir):
            logging.error("Invalid path: {0}".format(images_dir))
            sys.exit(1)

        # try:
        if os.path.exists(results_dir):
            FileHelper.remove_files_in_dir(results_dir)
        else:
            os.makedirs(results_dir)
        # except OSError:
        #     pass


        self.detector = cv2.SIFT(3200)
        # matcher = cv2.BFMatcher(cv2.NORM_L2)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.templatesFeatures = None

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

    def compareImageWithTemplates(self, image):

        self.templatesFeatures = self.__getFeaturesFromTemplates()
        kp2, desc2 = self.detector.detectAndCompute(image, None)
        logging.debug('img1 - %d features' % len(kp2))
        for name, (kp1, desc1) in self.templatesFeatures.iteritems():
            raw_matches = self.matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
            kp_pairs = self.__filter_matches(kp1, kp2, raw_matches)

            if len(kp_pairs) >= self.MIN_MATCH_COUNT:
                img1 = cv2.imread(os.path.join(self.templates_dir, name), cv2.IMREAD_GRAYSCALE)
                self.draw_matches(name, kp_pairs, img1, image)
                print name
            else:
                logging.debug("Not enough matches are found - %d/%d" % (len(kp_pairs), self.MIN_MATCH_COUNT))


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

    def draw_matches(self, window_name, kp_pairs, img1, img2):
        """Draws the matches"""
        mkp1, mkp2 = zip(*kp_pairs)

        H = None
        status = None

        if len(kp_pairs) >= 4:
            p1 = numpy.float32([kp.pt for kp in mkp1])
            p2 = numpy.float32([kp.pt for kp in mkp2])
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

        if len(kp_pairs):
            self.explore_match(window_name, img1, img2, kp_pairs, status, H)


    def explore_match(self, win, img1, img2, kp_pairs, status=None, H=None):
        """Draws lines between the matched features"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis = numpy.zeros((max(h1, h2), w1 + w2), numpy.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if H is not None:
            corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            reshaped = cv2.perspectiveTransform(corners.reshape(1, -1, 2), H)
            reshaped = reshaped.reshape(-1, 2)
            corners = numpy.int32(reshaped + (w1, 0))
            cv2.polylines(vis, [corners], True, (255, 255, 255))

        if status is None:
            status = numpy.ones(len(kp_pairs), numpy.bool_)
        p1 = numpy.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = numpy.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

        green = (0, 255, 0)
        red = (0, 0, 255)
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                col = green
                cv2.circle(vis, (x1, y1), 2, col, -1)
                cv2.circle(vis, (x2, y2), 2, col, -1)
            else:
                col = red
                r = 2
                thickness = 3
                cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
                cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
                cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
                cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
        vis0 = vis.copy()
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)

        cv2.imshow(win, vis)

###############################################################################
# Test Main
###############################################################################

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    # if len(sys.argv) < 3:
    #     print "No filenames specified"
    #     print "USAGE: find_obj.py <path_to_logos> <path_to_dir_with_images>"
    #     sys.exit(1)
    #
    # dir_logo_db = sys.argv[1]
    # dir_images = sys.argv[2]



    obj = SIFTComparer('../logos', '../in')
    # img1 = cv2.imread('../in/784_00493.jpg', cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.imread('../in/clip0a28ae_00004.jpg', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('../in/Pic21.png', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(dir_images, 0)

    obj.compareImageWithTemplates(img1)

    #
    # MIN_MATCH_COUNT = 2
    #
    # if img1 is None:
    #     print 'Failed to load fn1:', dir_logo_db
    #     sys.exit(1)
    #
    # if img2 is None:
    #     print 'Failed to load fn2:', dir_images
    #     sys.exit(1)
    #
    # kp_pairs = match_images(img1, img2)
    #
    # if len(kp_pairs) >= MIN_MATCH_COUNT:
    #     draw_matches('find_obj', kp_pairs, img1, img2)
    # else:
    #     print "Not enough matches are found - %d/%d" % (len(kp_pairs), MIN_MATCH_COUNT)
    #
    cv2.waitKey()
    cv2.destroyAllWindows()