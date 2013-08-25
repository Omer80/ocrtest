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

    def extractFeatures(self, positiveImageTemplate=None):
        windowSize, shiftSize, tagPosition = self.windowSize, self.shiftSize, self.tagPosition

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