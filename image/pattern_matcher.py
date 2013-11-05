import os
from heapq import nlargest, nsmallest

import cv2

from misc.file_helper import FileHelper


class PatternMatcher(object):

    def load_patterns(self, filelist):
        self.pattern_filelist = filelist
        self.patterns = []
        for filename in filelist:
            image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            self.patterns.append(image)

    def load_patterns_folder(self, folder):
        pattern_filelist = []
        for dir, filename in FileHelper.read_images_in_dir_recursively(folder):
            pattern_filelist.append(os.path.join(dir, filename))

        self.load_patterns(pattern_filelist)

    def match(self, filename, visualise=False):
        image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # globalMin, globalLoc = None, None
        # bestMatches = []

        methods = [
            # (cv2.TM_SQDIFF, True, [], [255, 0, 0], (+1, +1)),
            # (cv2.TM_SQDIFF_NORMED, True, [], [0, 255, 0], (-1, -1)),
            # (cv2.TM_CCOEFF, False, [], [0, 0, 255], (+2, 0)),
            (cv2.TM_CCOEFF_NORMED, False, [], [0, 255, 255], (0, 0))
        ]
        # methods = [
        #     (cv2.TM_SQDIFF, True, [], [255, 0, 0]),
        #     (cv2.TM_CCORR, False, [], [0, 255, 0]),
        #     (cv2.TM_CCOEFF, False, [], [0, 0, 255])
        # ]
        # methods = [
        #     (cv2.TM_SQDIFF_NORMED, True, [], [255, 0, 0]),
        #     (cv2.TM_CCORR_NORMED, False, [], [0, 255, 0]),
        #     (cv2.TM_CCOEFF_NORMED, False, [], [0, 0, 255])
        # ]
        # matchMethod, getMin = cv2.TM_SQDIFF_NORMED, True
        # matchMethod, getMin = cv2.TM_CCORR_NORMED, False
        # matchMethod, getMin = cv2.TM_CCOEFF_NORMED, False
        for pattern, pfn in zip(self.patterns, self.pattern_filelist):
            # print 'Process: ', pfn
            for matchMethod, getMin, bestMatches, _, _ in methods:
                result = cv2.matchTemplate(image, pattern, matchMethod)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
                # print 'minval: ', minVal, 'minloc: ', minLoc
                if getMin:
                    bestMatches.append((minVal, minLoc, pattern.shape))
                else:
                    bestMatches.append((maxVal, maxLoc, pattern.shape))
            # if globalMin is None or globalMin > minVal:
            #     # print 'best'
            #     print 'minval: ', minVal, 'minloc: ', minLoc
            #     globalMin = minVal
            #     globalLoc = minLoc
            #     bestPattern = pattern

        if visualise:
            imageShown = cv2.imread(filename)
            for _, getMin, bestMatches, c, shift in methods:
                color = list(c)
                if getMin:
                    best = nsmallest(10, bestMatches)
                else:
                    best = nlargest(10, bestMatches)
                print [e[0] for e in best]
                for val, loc, patternShape in best:
                    cv2.rectangle(imageShown, (loc[0]+shift[0], loc[1]+shift[1]), (loc[0]+patternShape[0]+shift[0], loc[1]+patternShape[1]+shift[1]), tuple(color), 1)
                    color = [127 if e == 0 else e for e in color]
                    # cv2.rectangle(image, (loc[0]+1, loc[1]+1), (loc[0]+pattern.shape[0]-1, loc[1]+pattern.shape[1]-1), (0, 0, 0), 1)

            cv2.imshow('image', imageShown)
            key = 255
            while key != 32:
                key = cv2.waitKey(20) & 0xFF
                # if key != 255:
                #     print key
                if key == 27:
                    return

        results = []
        for _, getMin, bestMatches, _, _ in methods:
            if getMin:
                best = nsmallest(10, bestMatches)
            else:
                best = nlargest(10, bestMatches)
            results.append(best)

        return results