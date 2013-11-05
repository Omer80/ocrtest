import os

import cv2
from sklearn.externals.joblib import delayed, Parallel

from classifier.windowed import WindowedFeatureClassifier
from dataset.utils import load
from image.pattern_matcher import PatternMatcher
from misc.file_helper import FileHelper
from ocr_utils import ArgParserWithDefaultHelp

from temp_scripts.check_windowed_classifier import ColorMap


def process_image(classifier, patternMatcher, filename, output):
    colorMap = ColorMap((0, 255, 255), (0, 0, 255), 0.0, 1.0)
    matchingRegions = patternMatcher.match(filename, visualise=False)[0]
    filename, result = classifier.process_file(filename)

    image = cv2.imread(filename)
    for val, loc, patternShape in matchingRegions[::-1]:
        if val >= 0.9:
            color = (255, 0, 0)
            shift = 0
        elif val >= 0.8:
            color = (0, 255, 0)
            shift = 1
        else:
            color = (255, 255, 0)
            shift = -1

        cv2.rectangle(image, (loc[0]+shift, loc[1]+shift), (loc[0]+patternShape[0]+shift, loc[1]+patternShape[1]+shift), color, 1)

    for r, (x1, y1, x2, y2) in result:
        cv2.rectangle(image, (y1, x1), (y2, x2), colorMap[r], 1)

    path, bf = os.path.split(filename)
    cv2.imwrite(os.path.join(output, bf), image)

if __name__ == '__main__':
    parser = ArgParserWithDefaultHelp(description='Visualising tool for pattern matcher and classifier')
    parser.add_argument('folder', help='Folder, that contains folders with frames')
    parser.add_argument('patterns', help='Folder with patterns')
    parser.add_argument('classifier', help='Classifier')
    parser.add_argument('output', help='Output folder for processed images')
    parser.add_argument('-j', '--jobs', default=-1, type=int, help='Processes amount for parallel processing')
    args = parser.parse_args()

    classifier = load(args.classifier)
    FileHelper.create_or_clear_dir(args.output)

    wfc = WindowedFeatureClassifier(classifier)

    pm = PatternMatcher()
    pm.load_patterns_folder(args.patterns)

    tasks = []
    for dir, filename in FileHelper.read_images_in_dir_recursively(args.folder):
        tasks.append(delayed(process_image)(wfc, pm, os.path.join(dir, filename), args.output))

    p = Parallel(n_jobs=args.jobs, verbose=100)
    p(tasks)
