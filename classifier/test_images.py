import os
import logging
import shutil

from skimage.draw._draw import line
from skimage.io import imsave
from sklearn.externals import joblib

from image.processing import Image
from misc.file_helper import FileHelper


def drawRectangle(image, rec):
    rr, cc = line(rec[0], rec[1], rec[0], rec[3])
    image[rr, cc] = 1
    rr, cc = line(rec[0], rec[3], rec[2], rec[3])
    image[rr, cc] = 1
    rr, cc = line(rec[2], rec[3], rec[2], rec[1])
    image[rr, cc] = 1
    rr, cc = line(rec[2], rec[1], rec[0], rec[1])
    image[rr, cc] = 1


def process_folder(classifier, inputFolder, countPositive, outputFolder=None):
    logger = logging.getLogger("TestClassifier")
    total, amount = 0, 0
    for filename in FileHelper.read_images_in_dir(inputFolder):
        total += 1
        image = Image(os.path.join(inputFolder, filename))
        _, windows = image.process()
        result = classifier.predict(windows)

        px, py = -1, -1
        isPositive = False
        for i, (w, r) in enumerate(zip(windows, result)):
            if r:
                xc = (i / image.windowsAmountInfo[1])
                yc = (i % image.windowsAmountInfo[1])
                if outputFolder:
                    x = xc * image.shiftSize[0]
                    y = yc * image.shiftSize[1]
                    b = image.bounds
                    drawRectangle(image.sourceImage, (x + b[0].start - image.missingRows,
                                                      y + b[1].start - image.missingColumns,
                                                      x+image.windowSize[0]-1 + b[0].start - image.missingRows,
                                                      y+image.windowSize[1]-1 + b[1].start - image.missingColumns)
                    )

                if xc == px and yc == py + 1:
                    isPositive = True
                px, py = xc, yc

        logger.debug('Processed %s (%s)' % (filename, 'positive' if isPositive else 'negative'))
        if isPositive == countPositive:
            amount += 1
            if outputFolder:
                imsave(os.path.join(outputFolder, filename), image.sourceImage)

    return total, amount


def process_sample(classifier, inputFolder, outputFolder=None):
    logger = logging.getLogger("TestClassifier")

    positiveInput = os.path.join(inputFolder, 'positive')
    negativeInput = os.path.join(inputFolder, 'negative')

    falseNegativeOutput, falsePositiveOutput = None, None
    if outputFolder:
        falsePositiveOutput = os.path.join(outputFolder, 'falsePositive')
        falseNegativeOutput = os.path.join(outputFolder, 'falseNegative')
        FileHelper.create_or_clear_dir(falsePositiveOutput)
        FileHelper.create_or_clear_dir(falseNegativeOutput)

    logger.debug('Process negative examples')
    trueNegative, falsePositive = process_folder(classifier, negativeInput, True, falsePositiveOutput)
    logger.info('False positives: %d; True negatives: %d' % (falsePositive, trueNegative))
    logger.debug('Process positive examples')
    truePositive, falseNegative = process_folder(classifier, positiveInput, False, falseNegativeOutput)
    logger.info('True positives: %d; False negatives: %d' % (truePositive, falseNegative))

    return truePositive, falseNegative, trueNegative, falsePositive


def loadClassifier(filename):
    classifier = joblib.load(filename)
    classifier.set_params(n_jobs=1)
    return classifier

