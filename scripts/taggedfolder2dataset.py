import argparse

from dataset.image_folder_group import process_base_folder, save_dataset


def process_arguments():
    parser = argparse.ArgumentParser(description='Dataset creation tool from several folders')
    parser.add_argument('folder', help='Folder, that contains folders with frames')
    parser.add_argument('train', help='Train dataset')
    parser.add_argument('test', help='Test dataset')
    parser.add_argument('--train_files', default=None, help='File with list of images, that included in train set')
    parser.add_argument('--test_files', default=None, help='File with list of images, that included in test set')
    parser.add_argument('-p', '--positive-fragments-folder', dest='positive_fragments_folder', default=None, help='Folder to put positive fragments of frames')
    parser.add_argument('-t', '--type', default='large', choices=['large', 'small'], help='Size of train set')
    parser.add_argument('-m', '--negmult', default=3, type=int, help='Negative multiplicator: how more negative examples than positive')
    parser.add_argument('-j', '--jobs', default=-1, type=int, help='Processes amount for feature extraction')
    parser.add_argument('-o', '--dataset-type', dest='dataset_type', default='pkl', choices=['pkl', 'csv'], help='Type of dataset output')
    parser.add_argument('--only-first-symbol', dest='first_symbol_tag', action='store_true', help='Positive only on first symbol from tag')
    parser.set_defaults(first_symbol_tag=False)
    # todo: process 'only first symbol' parameter

    return parser.parse_args()


if __name__ == '__main__':
    import ocr_utils
    ocr_utils.init_console_logging()

    args = process_arguments()
    dc = process_base_folder(args.folder,
                             negativeMultiplicator=args.negmult,
                             rulesType=args.type,
                             jobs=args.jobs,
                             interestingWindowsFolder=args.positive_fragments_folder,
                             onlyFirstTagSymbol=args.first_symbol_tag
    )
    save_dataset(dc, args.train, args.test, args.train_files, args.test_files, args.dataset_type == 'csv')