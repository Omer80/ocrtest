import os
import logging
import shutil
import numpy as np

from skimage.draw._draw import line
from skimage.io import imsave
from sklearn.externals import joblib

from process_image import Image
from sliding_window import sliding_window


def drawRectangle(image, rec):
    rr, cc = line(rec[0], rec[1], rec[0], rec[3])
    image[rr, cc] = 1
    rr, cc = line(rec[0], rec[3], rec[2], rec[3])
    image[rr, cc] = 1
    rr, cc = line(rec[2], rec[3], rec[2], rec[1])
    image[rr, cc] = 1
    rr, cc = line(rec[2], rec[1], rec[0], rec[1])
    image[rr, cc] = 1


def processDirectory(classifier, inputFolder, outputFolder=None):
    logger = logging.getLogger("TestClassifier")
    acceptableExtensions = ('jpg', 'jpeg', 'png')
    try:
        shutil.rmtree(os.path.join(outputFolder, 'positive'))
        shutil.rmtree(os.path.join(outputFolder, 'negative'))
        os.makedirs(os.path.join(outputFolder, 'positive'))
        os.makedirs(os.path.join(outputFolder, 'negative'))
    except OSError:
        pass

    for filename in os.listdir(inputFolder):
        if filename.endswith(acceptableExtensions):
            logger.debug('Processing %s' % (filename,))
            image = Image(os.path.join(inputFolder, filename), windowSize=(30, 30), shiftSize=(3, 3))
            # _, windows = image.process()
            image.prepare()
            print filename
            s = ((np.array(image.image.shape) - np.array(image.windowSize)) // np.array(image.shiftSize)) + 1
            print s
            image.windowsAmountInfo = s
            windows = sliding_window(image = image.image, windowSize=(30, 30),shiftSize=(3, 3), flatten=True)
            l = windows.shape
            windows = windows.reshape(l[0],900)
            windows = np.insert(windows, 0, 0, axis=1 )

            result = classifier.predict(windows)
            print np.sum(result)
            

            # out = os.path.join(outputFolder, 'negative')
            # px, py = -1, -1
            # for i, (w, r) in enumerate(zip(windows, result)):
            #     if r:
            #         xc = (i / image.windowsAmountInfo[1])
            #         yc = (i % image.windowsAmountInfo[1])
            #         x = xc * image.shiftSize[0]
            #         y = yc * image.shiftSize[1]
            #         b = image.bounds
            #         drawRectangle(image.sourceImage, (x + b[0].start,
            #                                           y + b[1].start,
            #                                           x+image.windowSize[0] + b[0].start,
            #                                           y+image.windowSize[1] + b[1].start)
            #         )
            #         if xc == px and yc == py + 1:
            #             out = os.path.join(outputFolder, 'positive')
            #         px, py = xc, yc
            #
            # imsave(os.path.join(out, filename), image.sourceImage)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print 'USAGE:\n\t' + sys.argv[0] + ' classifier.pkl folderWithImages outputFolder'
        sys.exit(1)

    classifier = joblib.load(sys.argv[1])
    processDirectory(classifier, sys.argv[2], sys.argv[3])