import argparse
import os

from create_dataset import DatasetCreator
from process_folder import process_folder, large_train, small_train


def process_base_folder(folder, negativeMultiplicator=3, rulesType='large', jobs=-1, prefix=None, interestingWindowsFolder=None):
    dc = DatasetCreator()
    for f in os.listdir(folder):
        ff = os.path.join(folder, f)
        if os.path.isdir(ff):
            if not prefix or f.startswith(prefix):
                rules = large_train if rulesType == 'large' else small_train
                process_folder(ff, rules=rules, negativeMultiplicator=negativeMultiplicator, datasetCreator=dc, interestingWindowsFolder=interestingWindowsFolder)

    dc.processPrepared(jobs)
    return dc


def save_dataset(dc, trainFilename, testFilename, trainImageFilenames, testImageFilenames):
    dc.saveCSV(trainFilename, testFilename)
    if trainImageFilenames and testImageFilenames:
        dc.saveTrainTestImageFilenames(trainImageFilenames, testImageFilenames)


def process_arguments():
    parser = argparse.ArgumentParser(description='Dataset creation tool from several folders')
    parser.add_argument('folder', help='Folder, that contains folders with frames')
    parser.add_argument('train', help='Train dataset')
    parser.add_argument('test', help='Test dataset')
    parser.add_argument('train_files', default=None, help='File with list of images, that included in train set')
    parser.add_argument('test_files', default=None, help='File with list of images, that included in test set')
    parser.add_argument('-p', '--positive_fragments_folder', default=None, help='Folder to put positive fragments of frames')
    parser.add_argument('-t', '--type', default='large', choices=['large', 'small'], help='Size of train set')
    parser.add_argument('-m', '--negmult', default=3, type=int, help='Negative multiplicator: how more negative examples than positive')
    parser.add_argument('-j', '--jobs', default=-1, type=int, help='Processes amount for feature extraction')

    return parser.parse_args()


if __name__ == '__main__':
    import utils
    utils.init_console_logging()

    # import sys
    # if len(sys.argv) < 4:
    #     print 'USAGE:\n\t' + sys.argv[0] + ' folder_with_framefolders train.csv test.csv [trainFiles.csv testFiles.csv]'
    #     sys.exit(1)
    #
    # if len(sys.argv) >= 6:
    #     trnfn = sys.argv[4]
    #     tstfn = sys.argv[5]
    # else:
    #     trnfn, tstfn = None, None

    args = process_arguments()
    dc = process_base_folder(args.folder, negativeMultiplicator=args.negmult, rulesType=args.type, jobs=args.jobs, interestingWindowsFolder=args.positive_fragments_folder)
    save_dataset(dc, args.train, args.test, args.train_files, args.test_files)