import os

import cv2
import numpy as np
# import matplotlib.pyplot as plt

from classifier.windowed import WindowedFeatureClassifier
from dataset.utils import load
from misc.file_helper import FileHelper


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


def process_file(wfc, filename, output):
    filename, result = wfc.process_file(filename)

    # evaluation = np.array([e[0] for e in result])

    # plt.hist(evaluation, bins=50, log=True)
    # plt.show()

    imageShowed = cv2.imread(filename)
    for r, (x1, y1, x2, y2) in result:
        cv2.rectangle(imageShowed, (y1, x1), (y2, x2), colorMap[r], 1)

    path, filename = os.path.split(filename)
    bfn, ext = os.path.splitext(filename)
    cv2.imwrite(os.path.join(output, bfn+'_wclrd'+ext), imageShowed)

    # cv2.imshow('image', imageShowed)
    # key = 255
    # while key != 27:
    #     key = cv2.waitKey(20) & 0xFF


if __name__ == '__main__':
    import sys
    classifier = load(sys.argv[1])
    output = sys.argv[3]
    FileHelper.create_or_clear_dir(output)

    colorMap = ColorMap((0, 255, 255), (0, 0, 255), 0.0, 1.0)
    wfc = WindowedFeatureClassifier(classifier)
    if os.path.isdir(sys.argv[2]):
        dir = sys.argv[2]
        for filename in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, filename)):
                process_file(wfc, os.path.join(dir, filename), output)
    else:
        process_file(wfc, sys.argv[2], output)
