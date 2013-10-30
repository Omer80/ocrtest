import random
import numpy as np
import logging

from scipy import ndimage
from skimage import feature
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.transform import resize

from image.window import sliding_window


class Image(object):
    def __init__(self, image, windowSize=(64, 64), shiftSize=(32, 32), tagPosition=None, saveFeatures=False):
        if isinstance(image, basestring):
            self.imagePath = image
            self.rawImage = None
        else:
            self.rawImage = image

        self.windowSize = windowSize
        self.shiftSize = shiftSize
        self.saveFeatures = saveFeatures

        # version for tagPosition creation with height and width instead of low-right corner coordinates
        # t = tagPosition
        # self.tagPosition = (t[0], t[1], t[0]+t[2], t[1]+t[2])

        self.tagPosition = tagPosition

        self.finalWindowResolution = (32, 32)

    def prepare(self):
        if self.rawImage is None:
            self.rawImage = imread(self.imagePath)
        self.sourceImage = rgb2gray(self.rawImage)
        # remove black borders from image
        iim = self.sourceImage > 0
        imageObjects = ndimage.find_objects(iim)
        if imageObjects is not None and len(imageObjects) > 0:
            self.bounds = imageObjects[0]
        else:
            self.bounds = [slice(*(0, self.sourceImage.shape[0]-1)), slice(*(0, self.sourceImage.shape[1]-1))]
        self.image = self.sourceImage[self.bounds[0], self.bounds[1]]
        # get new tag position after cutting the image
        if self.tagPosition:
            t, b = self.tagPosition, self.bounds
            self.tagPosition = (t[0] - b[0].start, t[1] - b[1].start, t[2] - b[0].start, t[3] - b[1].start)

        # extend image to be divisible by window shift
        imsh = self.image.shape
        missingRows = 0
        if imsh[0] % self.shiftSize[0] != 0:
            missingRows = self.shiftSize[0] - (imsh[0] % self.shiftSize[0])
            self.image = np.vstack([self.image, np.reshape(np.zeros(missingRows * imsh[1]), (missingRows, imsh[1]))])
            self.missingRows = missingRows
            if self.tagPosition:
                t = self.tagPosition
                self.tagPosition = (t[0] + missingRows, t[1], t[2] + missingRows, t[3])
        self.missingRows = missingRows

        imsh = self.image.shape
        missingColumns = 0
        if imsh[1] % self.shiftSize[1] != 0:
            missingColumns = self.shiftSize[1] - (imsh[1] % self.shiftSize[1])
            self.image = np.hstack([self.image, np.reshape(np.zeros(missingColumns * imsh[0]), (imsh[0], missingColumns))])
            self.missingColumns = missingColumns
            if self.tagPosition:
                t = self.tagPosition
                self.tagPosition = (t[0], t[1] + missingColumns, t[2], t[3] + missingColumns)
        self.missingColumns = missingColumns

    def isWindowInTagArea(self, x, y):
        if (x+self.windowSize[0] - self.tagPosition[0]) >= (self.windowSize[0] / 4) \
                    and (y+self.windowSize[1] - self.tagPosition[1]) >= (self.windowSize[1] / 4) \
                    and (self.tagPosition[2] - x) >= (self.windowSize[0] / 4) \
                    and (self.tagPosition[3] - y) >= (self.windowSize[1] / 4):
            return True
        else:
            return False

    def createNeighbourWindows(self, x, y, amount=7):
        coordinates = set()
        tries = 0
        while len(coordinates) < amount or tries < amount * 5:
            tries += 1
            nx = int(random.gauss(x, self.shiftSize[0] / 2))
            ny = int(random.gauss(y, self.shiftSize[1] / 2))
            if nx >= 0 and nx + self.windowSize[0] < self.image.shape[0] \
                    and ny >= 0 and ny + self.windowSize[1] < self.image.shape[1]\
                    and self.isWindowInTagArea(nx, ny):
                coordinates.add((nx, ny))

        return [self.getWindow(nx, ny) for nx, ny in coordinates]

    def getWindow(self, x, y):
        return self.image[x:x+self.windowSize[0], y:y+self.windowSize[1]]

    def extractAllFeatures(self, positiveImageTemplate=None):
        windowSize, shiftSize, tagPosition = self.windowSize, self.shiftSize, self.tagPosition
        # if positiveImageTemplate is not None:
        #     imsave(positiveImageTemplate % (-1,), self.image)

        # count rows/columns amount
        s = ((np.array(self.image.shape) - np.array(windowSize)) // np.array(shiftSize)) + 1
        self.windowsAmountInfo = s
        windows = sliding_window(self.image, windowSize, shiftSize)
        positiveExamples = []
        negativeExamples = []
        j = 0
        for i, w in enumerate(windows):
            x, y = (i / s[1])*shiftSize[0], (i % s[1])*shiftSize[1]

            wSized = resize(w, self.finalWindowResolution)
            features = feature.hog(wSized)

            if self.tagPosition and self.isWindowInTagArea(x, y):
                if positiveImageTemplate is not None:
                    imsave(positiveImageTemplate % (j,), w)
                    j += 1
                positiveExamples.append(features)
            else:
                negativeExamples.append(features)

        if self.saveFeatures:
            self.positiveExamples = positiveExamples
            self.negativeExamples = negativeExamples

        return positiveExamples, negativeExamples

    def extractFeatures(self, positiveImageTemplate=None, negativeMultiplicator=None, positiveWindowNeighboursAmount=7):
        if negativeMultiplicator is None or self.tagPosition is None:
            return self.extractAllFeatures(positiveImageTemplate)

        windowSize, shiftSize, tagPosition = self.windowSize, self.shiftSize, self.tagPosition

        # count rows/columns amount
        s = ((np.array(self.image.shape) - np.array(windowSize)) // np.array(shiftSize)) + 1
        self.windowsAmountInfo = s

        negativeIndexes = []
        positiveWindows = []

        windowsAmount = s[0] * s[1]
        x, y = -shiftSize[0], 0
        for i in xrange(windowsAmount):
            y += shiftSize[1]
            if i % s[1] == 0:
                y = 0
                x += shiftSize[0]

            if self.isWindowInTagArea(x, y):
                positiveWindows.append(self.getWindow(x, y))
                positiveWindows.extend(self.createNeighbourWindows(x, y, amount=positiveWindowNeighboursAmount))
            else:
                negativeIndexes.append(i)

        negativeAmount = int(len(positiveWindows) * negativeMultiplicator)
        if negativeAmount < len(negativeIndexes):
            random.shuffle(negativeIndexes)
            negativeIndexes = negativeIndexes[:negativeAmount]

        negativeWindows = []
        for i in negativeIndexes:
            x, y = (i / s[1])*shiftSize[0], (i % s[1])*shiftSize[1]
            negativeWindows.append(self.getWindow(x, y))

        j = 0
        positiveExamples = []
        for window in positiveWindows:
            wSized = resize(window, self.finalWindowResolution)
            features = feature.hog(wSized)
            positiveExamples.append(features)

            if positiveImageTemplate is not None:
                try:
                    imsave(positiveImageTemplate % (j,), window)
                    j += 1
                except ValueError, e:
                    print self.imagePath
                    print e

        negativeExamples = []
        for window in negativeWindows:
            wSized = resize(window, self.finalWindowResolution)
            features = feature.hog(wSized)
            negativeExamples.append(features)

        if self.saveFeatures:
            self.positiveExamples = positiveExamples
            self.negativeExamples = negativeExamples

        return positiveExamples, negativeExamples

    def process(self, positiveImageTemplate=None, negativeMultiplicator=None, positiveWindowNeighboursAmount=7):
        self.prepare()
        return self.extractFeatures(positiveImageTemplate, negativeMultiplicator, positiveWindowNeighboursAmount)


def process_single_image(filename, tagPosition, positiveImageTemplate=None, negativeMultiplicator=None, positiveWindowNeighboursAmount=7):
    image = Image(filename, tagPosition=tagPosition)
    return image.process(positiveImageTemplate, negativeMultiplicator, positiveWindowNeighboursAmount)

