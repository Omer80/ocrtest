import os

import cv2
import numpy as np
# import matplotlib.pyplot as plt

from classifier.windowed import WindowedFeatureClassifier
from dataset.utils import load


class ColorMap(object):
    startcolor = ()
    endcolor = ()
    startmap = 0
    endmap = 0
    colordistance = 0
    valuerange = 0
    ratios = []

    def __init__(self, startcolor, endcolor, startmap, endmap):
        self.startcolor = np.array(startcolor)
        self.endcolor = np.array(endcolor)
        self.startmap = float(startmap)
        self.endmap = float(endmap)
        self.valuerange = float(endmap - startmap)
        self.ratios = (self.endcolor - self.startcolor) / self.valuerange

    def __getitem__(self, value):
        color = tuple(self.startcolor + (self.ratios * (value - self.startmap)))
        return int(color[0]), int(color[1]), int(color[2])


if __name__ == '__main__':
    import sys
    classifier = load(sys.argv[1])

    colorMap = ColorMap((0, 255, 255), (0, 0, 255), 0.0, 1.0)
    wfc = WindowedFeatureClassifier(classifier)
    filename, result = wfc.process_file(sys.argv[2])

    # evaluation = np.array([e[0] for e in result])

    # plt.hist(evaluation, bins=50, log=True)
    # plt.show()

    imageShowed = cv2.imread(sys.argv[2])
    for r, (x1, y1, x2, y2) in result:
        cv2.rectangle(imageShowed, (y1, x1), (y2, x2), colorMap[r], 1)

    path, filename = os.path.split(sys.argv[2])
    bfn, ext = os.path.splitext(filename)
    cv2.imwrite(os.path.join(sys.argv[3], bfn+'_wclrd'+ext), imageShowed)

    # cv2.imshow('image', imageShowed)
    # key = 255
    # while key != 27:
    #     key = cv2.waitKey(20) & 0xFF
