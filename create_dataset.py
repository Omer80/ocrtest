import csv
import os
import logging
import random

import numpy as np

from process_image import Image


class DatasetCreator(object):
    def __init__(self, testPart=0.3, negativeMultiplicator=3, seed=8172635):
        self.trainDataset = []
        self.testDataset = []
        self.testPart = testPart
        self.negativeMultiplicator = negativeMultiplicator
        self.seed = seed
        self.rand = random.Random()
        self.rand.seed(seed)

    def set_image_folder(self, imageFolder, interestingWindowsFolder=None):
        self.imageFolder = imageFolder
        self.interestingWindowsFolder = interestingWindowsFolder
        if interestingWindowsFolder and not os.path.exists(interestingWindowsFolder):
            os.makedirs(interestingWindowsFolder)
        self.tagPosition = None
        tagInformationParts = imageFolder.rsplit('_', 1)
        if len(tagInformationParts) > 1:
            tagInformationString = tagInformationParts[1]
            tagCoords = tagInformationString.split('x')
            if len(tagCoords) == 4:
                self.tagPosition = map(int, tagCoords)

        if self.tagPosition is None:
            raise ValueError("Incorrect folder name format. Folder MUST contain tag position information")

    def directoryProcess(self, imageFolder, interestingWindowsFolder=None):
        self.set_image_folder(imageFolder, interestingWindowsFolder)

        logger = logging.getLogger("RawFeatureExtractor")
        acceptableExtensions = ('jpg', 'jpeg', 'png')
        appropriateFiles = []
        for filename in os.listdir(self.imageFolder):
            if filename.endswith(acceptableExtensions):
                appropriateFiles.append(filename)
        self.rand.shuffle(appropriateFiles)
        ttSplitIndx = int(len(appropriateFiles) * (1-self.testPart))
        trainFiles = set(appropriateFiles[:ttSplitIndx])
        testFiles = set(appropriateFiles[ttSplitIndx:])
        appropriateFilesSet = set(appropriateFiles)

        for filename in os.listdir(self.imageFolder):
            if filename not in appropriateFilesSet:
                continue

            logger.debug('Processing %s' % (filename,))
            if self.interestingWindowsFolder:
                name, extension = os.path.splitext(filename)
                positiveImageTemplate = os.path.join(self.interestingWindowsFolder, name + '_%d' + extension)
            else:
                positiveImageTemplate = None

            image = Image(os.path.join(self.imageFolder, filename), tagPosition=self.tagPosition)
            positive, negative = image.process(positiveImageTemplate=positiveImageTemplate)
            if len(positive) == 0:
                logger.warning('No positive windows were created in image: %s' % (filename,))

            self.rand.shuffle(negative)
            negativeMAmount = int(len(positive) * self.negativeMultiplicator)
            if negativeMAmount < len(negative):
                negative = negative[:negativeMAmount]

            if filename in trainFiles:
                dataset = self.trainDataset
            else:
                dataset = self.testDataset

            for e in positive:
                dataset.append(np.concatenate([e, np.array([1])]))
            for e in negative:
                dataset.append(np.concatenate([e, np.array([0])]))

    def saveCSV(self, trainFilename, testFilename):
        self.rand.shuffle(self.trainDataset)
        self.rand.shuffle(self.testDataset)

        with open(trainFilename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(self.trainDataset)

        with open(testFilename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(self.testDataset)

if __name__ == '__main__':
    import utils
    utils.init_console_logging()

    import sys
    if len(sys.argv) < 4:
        print 'USAGE:\n\t' + sys.argv[0] + ' folderWithImages train.csv test.csv'
        print '\nfolderWithImages name format: folderName_X1xY1xX2xY2, where X1xY1xX2xY2 coordinates of rectangle with hashtag'
        sys.exit(1)

    d = DatasetCreator(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[1]) + '_interesting')
    d.directoryProcess()
    d.saveCSV(sys.argv[2], sys.argv[3])
