from scipy import ndimage
from skimage import feature
from skimage.color import rgb2gray
from skimage.io import imread, imsave

import numpy as np
from skimage.transform import resize

from sliding_window import sliding_window


class Image(object):
    def __init__(self, imagePath, windowSize=(64, 64), shiftSize=(32, 32), tagPosition=None):
        self.imagePath = imagePath
        self.windowSize = windowSize
        self.shiftSize = shiftSize

        # version for tagPosition creation with height and width instead of low-right corner coordinates
        # t = tagPosition
        # self.tagPosition = (t[0], t[1], t[0]+t[2], t[1]+t[2])

        self.tagPosition = tagPosition

        self.finalWindowResolution = (32, 32)

    def prepare(self):
        self.sourceImage = rgb2gray(imread(self.imagePath))
        # remove black borders from image
        iim = self.sourceImage > 0
        self.bounds = ndimage.find_objects(iim)[0]
        self.image = self.sourceImage[self.bounds[0], self.bounds[1]]
        # get new tag position after cutting the image
        if self.tagPosition:
            t, b = self.tagPosition, self.bounds
            self.tagPosition = (t[0] - b[0].start, t[1] - b[1].start, t[2] - b[0].start, t[3] - b[1].start)

        # extend image to be divisible by window shift
        imsh = self.image.shape
        self.missingRows = 0
        if imsh[0] % self.shiftSize[0] != 0:
            missingRows = self.shiftSize[0] - (imsh[0] % self.shiftSize[0])
            self.image = np.vstack([np.reshape(np.zeros(missingRows * imsh[1]), (missingRows, imsh[1])), self.image])
            self.missingRows = missingRows
            if self.tagPosition:
                t = self.tagPosition
                self.tagPosition = (t[0] + missingRows, t[1], t[2] + missingRows, t[3])

        imsh = self.image.shape
        self.missingColumns = 0
        if imsh[1] % self.shiftSize[1] != 0:
            missingColumns = self.shiftSize[1] - (imsh[1] % self.shiftSize[1])
            self.image = np.hstack([np.reshape(np.zeros(missingColumns * imsh[0]), (imsh[0], missingColumns)), self.image])
            self.missingColumns = missingColumns
            if self.tagPosition:
                t = self.tagPosition
                self.tagPosition = (t[0], t[1] + missingColumns, t[2], t[3] + missingColumns)

    def extractFeatures(self, positiveImageTemplate=None):
        windowSize, shiftSize, tagPosition = self.windowSize, self.shiftSize, self.tagPosition
        # if positiveImageTemplate is not None:
        #     imsave(positiveImageTemplate % (-1,), self.image)

        # count rows/columns amount
        s = ((np.array(self.image.shape) - np.array(windowSize)) // np.array(shiftSize)) + 1
        self.windowsAmountInfo = s
        windows = sliding_window(self.image, windowSize, shiftSize)
        self.positiveExamples = []
        self.negativeExamples = []
        j = 0
        for i, w in enumerate(windows):
            x, y = (i / s[1])*shiftSize[0], (i % s[1])*shiftSize[1]

            wSized = resize(w, self.finalWindowResolution)
            features = feature.hog(wSized)

            if self.tagPosition \
                    and (x+windowSize[0] - tagPosition[0]) >= (windowSize[0] / 3) \
                    and (y+windowSize[1] - tagPosition[1]) >= (windowSize[1] / 3) \
                    and (tagPosition[2] - x) >= (windowSize[0] / 3) \
                    and (tagPosition[3] - y) >= (windowSize[1] / 3):

                if positiveImageTemplate is not None:
                    imsave(positiveImageTemplate % (j,), w)
                    j += 1
                self.positiveExamples.append(features)
            else:
                self.negativeExamples.append(features)

    def process(self, positiveImageTemplate=None):
        self.prepare()
        self.extractFeatures(positiveImageTemplate)
        return self.positiveExamples, self.negativeExamples


def process_single_image(filename, tagPosition, positiveImageTemplate=None):
    image = Image(filename, tagPosition=tagPosition)
    return image.process(positiveImageTemplate=positiveImageTemplate)


if __name__ == '__main__':
    testDATALINEfilename = '5_07000.jpg'
    i = Image(testDATALINEfilename, tagPosition=(437, 488, 453, 581))
    i.process()

    # imsave('5_07000_ngr.jpg', i.image)