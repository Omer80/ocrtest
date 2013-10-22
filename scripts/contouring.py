import os

import cv2
import numpy as np
from misc.file_helper import FileHelper

from scripts import binarization


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


def draw_one_contour(image, contour, color=None):
    if color is None:
        color_shape = (0, 255, 0)
        color_box = (0, 0, 255)
    else:
        color_box = color_shape = color

    cv2.drawContours(image, [contour], -1, color_shape, 1)

    rect = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color_box, 2)


def contourSolidity(contour):
    """ Solidity is the ratio of contour area to its convex hull area."""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
    return solidity


def contourExtent(contour):
    """Extent is the ratio of contour area to bounding rectangle area."""
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = float(area) / rect_area
    return extent


def containsDecision(matchingCoef, contourSize, solidity, extent, template_contourSize, template_solidity, template_extent):
    if matchingCoef > 0.5:
        return False
    if abs(template_solidity - solidity) > 0.1 or abs(template_extent - extent) > 0.1:
        return False
    if contourSize > template_contourSize*5:
        return False
    return True


def contour_match(template, contours, image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    bestCoef = None
    bestContour = None
    matched = []
    for contour in contours:
        matchingCoef = cv2.matchShapes(template, contour, 1, 1.0)
        if matchingCoef > 0.0:
            matched.append((matchingCoef, contour))
        if matchingCoef > 0.0 and (matchingCoef < bestCoef or bestCoef is None):
            bestCoef = matchingCoef
            bestContour = contour
        # print matchingCoef

    bestResults = sorted(matched, key=lambda x: x[0])

    # draw_one_contour(image, bestContour)
    if bestResults > 2:
        draw_one_contour(image, bestResults[0][1], (255, 0, 0))
        draw_one_contour(image, bestResults[1][1], (0, 255))
        draw_one_contour(image, bestResults[2][1], (0, 0, 255))
    print 'Best:', bestCoef
    print 'Best contour length: ', len(bestContour)

    for res in bestResults[:3]:
        contour = res[1]
        print '%0.3f %3d %0.2f %0.2f ' % (res[0], len(contour), contourSolidity(contour), contourExtent(contour))
        # print 'Area: ', cv2.contourArea(contour)
        # print 'Perimeter: ', cv2.arcLength(contour, True)
        # print 'Solidity: ', contourSolidity(contour)

    return image


def approxContour(contour):
    return cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)


def approxAllContours(contours):
    result = []
    for contour in contours:
        result.append(approxContour(contour))

    return result

if __name__ == '__main__':
    # findContourMethod = cv2.CHAIN_APPROX_NONE
    findContourMethod = cv2.CHAIN_APPROX_SIMPLE

    basefile = '/home/valeriy/projects/hashtag/logos/frames/5_00450.jpg'
    basefile_without_ext = '/home/valeriy/projects/hashtag/logos/bin_frames/5_00450'

    # basefile = '/home/valeriy/projects/hashtag/logos/bin_frames/5_00735.jpg'
    # basefile_without_ext = '/home/valeriy/projects/hashtag/logos/bin_frames/5_00735'

    template = binarization.load_image_thresholding('/home/valeriy/projects/hashtag/logos/learn/twitter_big_template.png', 127)
    base_template = template.copy()
    template_contours, template_hierarchy = cv2.findContours(template, cv2.RETR_EXTERNAL, findContourMethod)

    base_template = drawContours(base_template, template_contours)
    cv2.imwrite('/home/valeriy/projects/hashtag/logos/learn/twitter_big_template_contours.png', base_template)

    if len(template_contours) > 1:
        print 'Warning: too complex template'

    template_contours = template_contours[0]
    # print len(template_contours)
    print '%0.3f %3d %0.2f %0.2f ' % (0.0, len(template_contours), contourSolidity(template_contours), contourExtent(template_contours))

    # template_contours = approxContour(template_contours)

    inFolder = '/home/valeriy/projects/hashtag/logos/twitter_frames/'
    outFolder = '/home/valeriy/projects/hashtag/logos/bin_twitter_frames/'
    FileHelper.create_or_clear_dir(outFolder)
    for filename in sorted(FileHelper.read_images_in_dir(inFolder))[:3]:
        print filename
        basefile = os.path.join(inFolder, filename)
        basefile_without_ext = os.path.join(outFolder, os.path.splitext(filename)[0])

        # image = binarization.load_image_adaptive(basefile, 11, 10, inverse=False)
        image = binarization.load_image_thresholding(basefile, 180)
        # image = binarization.load_image_thresholding(basefile, 200, binary_type=cv2.THRESH_BINARY_INV)
        contour_image = image.copy()
        contours, hierarchy = cv2.findContours(contour_image, cv2.RETR_TREE, findContourMethod)
        image_drawed = drawContours(image, contours)
        # cv2.imwrite(basefile_without_ext + '_bin_thresh_100.jpg', image_drawed)
        # cv2.imwrite(basefile_without_ext + '_contour.jpg', contour_image)

        # contours = approxAllContours(contours)
        image = contour_match(template_contours, contours, image)
        cv2.imwrite(basefile_without_ext + '_interesting_contours.jpg', image)
