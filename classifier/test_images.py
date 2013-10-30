import os
import logging

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


def process_image(classifier, image, positiveOutputFolder=None, negativeOutputFolder=None):
    _, windows = image.process()
    result = classifier.predict(windows)

    px, py = -1, -1
    isPositive = False
    for i, (w, r) in enumerate(zip(windows, result)):
        if r:
            xc = (i / image.windowsAmountInfo[1])
            yc = (i % image.windowsAmountInfo[1])
            if positiveOutputFolder or negativeOutputFolder:
                x = xc * image.shiftSize[0]
                y = yc * image.shiftSize[1]
                b = image.bounds
                drawRectangle(image.sourceImage, (x + b[0].start-1,  # - image.missingRows,
                                                  y + b[1].start-1,  # - image.missingColumns,
                                                  x+image.windowSize[0]-1 + b[0].start-1,  # - image.missingRows,
                                                  y+image.windowSize[1]-1 + b[1].start-1  # - image.missingColumns
                                                )
                )

            if xc == px and yc == py + 1:
                isPositive = True
            px, py = xc, yc

    if isPositive and positiveOutputFolder:
        imsave(os.path.join(positiveOutputFolder, os.path.split(image.imagePath)[1]), image.sourceImage)
    if not isPositive and negativeOutputFolder:
        imsave(os.path.join(negativeOutputFolder, os.path.split(image.imagePath)[1]), image.sourceImage)

    return image.imagePath, isPositive


def process_file_list(classifier, filelist, countPositive, positiveOutputFolder=None, negativeOutputFolder=None, jobs=-1):
    total, amount = 0, 0
    tasks = []
    for filename in filelist:
        total += 1
        image = Image(filename)
        tasks.append(delayed(process_image)(classifier, image, positiveOutputFolder, negativeOutputFolder))

    p = Parallel(n_jobs=jobs, verbose=100)
    results = p(tasks)

    for filename, isPositive in results:
        if isPositive == countPositive:
            amount += 1

    return total-amount, amount


def process_folder(classifier, inputFolder, countPositive, positiveOutputFolder=None, negativeOutputFolder=None, jobs=-1):
    filelist = [os.path.join(inputFolder, fn) for fn in FileHelper.read_images_in_dir(inputFolder)]
    return process_file_list(classifier, filelist, countPositive, positiveOutputFolder, negativeOutputFolder, jobs)


def process_sample(classifier, inputFolder, outputFolder=None, jobs=-1, saveCorrects=False):
    logger = logging.getLogger("TestClassifier")

    positiveInput = os.path.join(inputFolder, 'positive')
    negativeInput = os.path.join(inputFolder, 'negative')

    falseNegativeOutput, falsePositiveOutput = None, None
    trueNegativeOutput, truePositiveOutput = None, None
    if outputFolder:
        falsePositiveOutput = os.path.join(outputFolder, 'falsePositive')
        falseNegativeOutput = os.path.join(outputFolder, 'falseNegative')
        FileHelper.create_or_clear_dir(falsePositiveOutput)
        FileHelper.create_or_clear_dir(falseNegativeOutput)
        if saveCorrects:
            truePositiveOutput = os.path.join(outputFolder, 'truePositive')
            trueNegativeOutput = os.path.join(outputFolder, 'trueNegative')
            FileHelper.create_or_clear_dir(truePositiveOutput)
            FileHelper.create_or_clear_dir(trueNegativeOutput)

    logger.debug('Process positive examples')
    truePositive, falseNegative = process_folder(classifier, positiveInput, False, truePositiveOutput, falseNegativeOutput, jobs)
    logger.info('True positives: %d; False negatives: %d' % (truePositive, falseNegative))
    logger.debug('Process negative examples')
    trueNegative, falsePositive = process_folder(classifier, negativeInput, True, falsePositiveOutput, trueNegativeOutput, jobs)
    logger.info('False positives: %d; True negatives: %d' % (falsePositive, trueNegative))

    return truePositive, falseNegative, trueNegative, falsePositive


def loadClassifier(filename):
    classifier = joblib.load(filename)
    classifier.set_params(n_jobs=1)
    return classifier

