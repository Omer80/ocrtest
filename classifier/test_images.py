import os
import logging
import random

from skimage.draw._draw import line
from skimage.io import imsave
from sklearn.externals import joblib
from sklearn.externals.joblib import delayed, Parallel

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


def createNeighbourWindows(image, x, y, amount=7):
    coordinates = set()
    while len(coordinates) < amount:
        nx = int(random.gauss(x, image.shiftSize[0] / 2))
        ny = int(random.gauss(y, image.shiftSize[1] / 2))
        if nx + image.windowSize[0] < image.image.shape[0] and ny + image.windowSize[1] < image.image.shape[1]:
            coordinates.add((nx, ny))

    return [image.getWindow(nx, ny) for nx, ny in coordinates]


def process_image(classifier, image, outputFolder=None):
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

    if outputFolder:
        imsave(os.path.join(outputFolder, os.path.split(image.imagePath)[1]), image.sourceImage)

    return image.imagePath, isPositive


def process_file_list(classifier, filelist, countPositive, outputFolder=None, jobs=-1):
    total, amount = 0, 0
    tasks = []
    for filename in filelist:
        total += 1
        image = Image(filename)
        tasks.append(delayed(process_image)(classifier, image, outputFolder is not None))


    p = Parallel(n_jobs=jobs, verbose=100)
    results = p(tasks)

    for filename, isPositive in results:
        if isPositive == countPositive:
            amount += 1

    return total-amount, amount


def process_folder(classifier, inputFolder, countPositive, outputFolder=None, jobs=-1):
    filelist = [os.path.join(inputFolder, fn) for fn in FileHelper.read_images_in_dir(inputFolder)]
    return process_file_list(classifier, filelist, countPositive, outputFolder, jobs)


def process_sample(classifier, inputFolder, outputFolder=None, jobs=-1):
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
    trueNegative, falsePositive = process_folder(classifier, negativeInput, True, falsePositiveOutput, jobs)
    logger.info('False positives: %d; True negatives: %d' % (falsePositive, trueNegative))
    logger.debug('Process positive examples')
    truePositive, falseNegative = process_folder(classifier, positiveInput, False, falseNegativeOutput, jobs)
    logger.info('True positives: %d; False negatives: %d' % (truePositive, falseNegative))

    return truePositive, falseNegative, trueNegative, falsePositive


def loadClassifier(filename):
    classifier = joblib.load(filename)
    classifier.set_params(n_jobs=1)
    return classifier

