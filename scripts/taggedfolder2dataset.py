
from image.processing import setup_image_factory
from dataset.image_folder_group import process_base_folder, save_dataset
from ocr_utils import ArgParserWithDefaultHelp


def process_arguments():
    parser = ArgParserWithDefaultHelp(description='Dataset creation tool from several folders')
    parser.add_argument('folder', help='Folder, that contains folders with frames')
    parser.add_argument('train', help='Train dataset')
    parser.add_argument('test', help='Test dataset')
    parser.add_argument('--train_files', default=None, help='File with list of images, that included in train set')
    parser.add_argument('--test_files', default=None, help='File with list of images, that included in test set')
    parser.add_argument('-p', '--positive-fragments-folder', dest='positive_fragments_folder', default=None, help='Folder to put positive fragments of frames')
    parser.add_argument('-t', '--type', default='large', choices=['large', 'small'], help='Size of train set')
    parser.add_argument('-m', '--negmult', default=3, type=int, help='Negative multiplicator: how more negative examples than positive')
    parser.add_argument('-n', '--neighbours-for-positive', dest='neighbours', default=7, type=int, help='Generate this neighbours amount for every positive window')
    parser.add_argument('-j', '--jobs', default=-1, type=int, help='Processes amount for feature extraction')
    parser.add_argument('-o', '--dataset-type', dest='dataset_type', default='csv', choices=['pkl', 'csv'], help='Type of dataset output')
    parser.add_argument('-w', '--window-size', dest='window_size', default=64, type=int, help='Window size')
    parser.add_argument('-s', '--shift-size', dest='shift_size', default=32, type=int, help='Shift size')
    parser.add_argument('-r', '--features-window-size', dest='features_window_size', default=32, type=int, help='Features window size')
    parser.add_argument('-f', '--features-type', dest='features_type', default='hog', choices=['hog', 'daisy'], help='Features detector')
    parser.add_argument('--only-first-symbol', dest='first_symbol_tag', action='store_true', help='Positive only on first symbol from tag')
    parser.set_defaults(first_symbol_tag=False)

    return parser.parse_args()


if __name__ == '__main__':
    import ocr_utils
    ocr_utils.init_console_logging()

    args = process_arguments()

    setup_image_factory((args.window_size, args.window_size),
                        (args.shift_size, args.shift_size),
                        (args.features_window_size, args.features_window_size),
                        args.features_type,
                        False)

    dc = process_base_folder(args.folder,
                             negativeMultiplicator=args.negmult,
                             rulesType=args.type,
                             jobs=args.jobs,
                             interestingWindowsFolder=args.positive_fragments_folder,
                             onlyFirstTagSymbol=args.first_symbol_tag,
                             positiveWindowNeighboursAmount=args.neighbours)

    save_dataset(dc, args.train, args.test, args.train_files, args.test_files, args.dataset_type == 'csv')