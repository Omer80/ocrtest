import csv
import os
import logging

from process_image import Image


class RawFeaturesExtractor(object):
    def __init__(self, imageFolder, interestingWindowsFolder=None):
        self.positiveExamples = []
        self.negativeExamples = []
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

    def directoryProcess(self):
        logger = logging.getLogger("RawFeatureExtractor")
        acceptableExtensions = ('jpg', 'jpeg', 'png')
        for filename in os.listdir(self.imageFolder):
            if filename.endswith(acceptableExtensions):
                logger.debug('Processing %s' % (filename,))
                if self.interestingWindowsFolder:
                    name, extension = os.path.splitext(filename)
                    positiveImageTemplate = os.path.join(self.interestingWindowsFolder, name + '_%d' + extension)
                else:
                    positiveImageTemplate = None
                image = Image(os.path.join(self.imageFolder, filename), tagPosition=self.tagPosition)
                positive, negative = image.process(positiveImageTemplate=positiveImageTemplate)
                if self.tagPosition and len(positive) == 0:
                    logger.warning('No positive windows were created in image: %s' % (filename,))
                self.positiveExamples.extend(positive)
                self.negativeExamples.extend(negative)

    def saveCSV(self, positiveFilename, negativeFilename):
        with open(positiveFilename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(self.positiveExamples)

        with open(negativeFilename, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(self.negativeExamples)

if __name__ == '__main__':
    import utils
    utils.init_console_logging()

    import sys
    if len(sys.argv) < 4:
        print 'USAGE:\n\t' + sys.argv[0] + ' folderWithImages positive.csv negative.csv'
        print '\nfolderWithImages name format: folderName_X1xY1xX2xY2, where X1xY1xX2xY2 coordinates of rectangle with hashtag'
        sys.exit(1)

    d = RawFeaturesExtractor(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[1]) + '_interesting')
    d.directoryProcess()
    d.saveCSV(sys.argv[2], sys.argv[3])
