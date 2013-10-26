import os

from dataset.creator import DatasetCreator
from dataset.image_folder import process_folder, large_train, small_train


def process_base_folder(folder,
                        negativeMultiplicator=3,
                        rulesType='large',
                        jobs=-1,
                        prefix=None,
                        interestingWindowsFolder=None,
                        onlyFirstTagSymbol=False):

    dc = DatasetCreator()
    rules = large_train if rulesType == 'large' else small_train
    for f in os.listdir(folder):
        ff = os.path.join(folder, f)
        if os.path.isdir(ff):
            if not prefix or f.startswith(prefix):
                process_folder(ff,
                               rules=rules,
                               negativeMultiplicator=negativeMultiplicator,
                               datasetCreator=dc,
                               interestingWindowsFolder=interestingWindowsFolder,
                               onlyFirstTagSymbol=onlyFirstTagSymbol
                )

    dc.processPrepared(jobs)
    return dc


def save_dataset(datasetCreator, trainFilename, testFilename, trainImageFilenames, testImageFilenames, saveCSV=False):
    if saveCSV:
        datasetCreator.saveCSV(trainFilename, testFilename)
    else:
        datasetCreator.savePKL(trainFilename, testFilename)

    if trainImageFilenames and testImageFilenames:
        datasetCreator.saveTrainTestImageFilenames(trainImageFilenames, testImageFilenames)
