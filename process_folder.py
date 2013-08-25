import os
from collections import namedtuple
import random
from create_dataset import DatasetCreator

Condition = namedtuple('Condition', ['cond_function', 'train'])

large_train = [
    Condition(lambda x: x > 0, 0.7)
]
small_train = [
    Condition(lambda x: x <= 10, 0.7),
    Condition(lambda x: x < 100, 10),
    Condition(lambda x: x >= 100, 0.1)
]


def getTagCoordinates(folder, YX=True):
    tagPosition = None
    tagInformationParts = folder.rsplit('_', 1)
    if len(tagInformationParts) > 1:
        tagInformationString = tagInformationParts[1]
        tagCoords = tagInformationString.split('x')
        if len(tagCoords) == 4:
            tagPosition = map(int, tagCoords)
            if YX:
                tp = tagPosition
                tagPosition = (tp[1], tp[0], tp[3], tp[2])

    if tagPosition is None:
        raise ValueError("Incorrect folder name format. Folder MUST contain tag position information")

    return tagPosition


def process_folder(folder, rules=large_train, negativeMultiplicator=3, interestingWindowsFolder=None, datasetCreator=None):
    files = []
    acceptableExtensions = ('jpg', 'jpeg', 'png')
    for filename in os.listdir(folder):
        if filename.lower().endswith(acceptableExtensions):
            files.append(os.path.join(folder, filename))

    trainAmount = 0
    for c in rules:
        if c.cond_function(len(files)):
            if c.train >= 1:
                trainAmount = c.train
            else:
                trainAmount = int(len(files) * c.train)
            break

    # testAmount = len(files) - trainAmount
    # if testAmount <= 0:
    #     testAmount = 0

    random.shuffle(files)
    trainFiles = set(files[:trainAmount])
    testFiles = set(files[trainAmount:])

    tagPosition = getTagCoordinates(folder)

    if datasetCreator is None:
        datasetCreator = DatasetCreator()

    datasetCreator.prepareImageProcessing(trainFiles, testFiles, tagPosition, negativeMultiplicator, interestingWindowsFolder)

    return datasetCreator
