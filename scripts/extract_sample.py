"""
Extract sample from big directories with images
"""

import os
import shutil
import random

from collections import namedtuple

from misc.file_helper import FileHelper
from ocr_utils import ArgParserWithDefaultHelp

Condition = namedtuple('Condition', ['cond_function', 'train'])

middle_sample = [
    Condition(lambda x: x <= 10, 0.99),
    Condition(lambda x: x < 100, 10),
    Condition(lambda x: x >= 100, 15)
]

small_sample = [
    Condition(lambda x: x <= 4, 0.99),
    Condition(lambda x: x <= 10, 3),
    Condition(lambda x: x < 100, 5),
    Condition(lambda x: x >= 100, 7)
]

extra_small_sample = [
    Condition(lambda x: x <= 3, 0.99),
    Condition(lambda x: x <= 10, 3),
    Condition(lambda x: x < 100, 4),
    Condition(lambda x: x >= 100, 5)
]

default_sample = middle_sample

def get_ruled_files(files, rules):
    trainAmount = 0
    for c in rules:
        if c.cond_function(len(files)):
            if c.train >= 1:
                trainAmount = c.train
            else:
                trainAmount = int(len(files) * c.train)
            break

    random.shuffle(files)
    return files[:trainAmount]


def process_folder(folder, rules=default_sample):
    files = FileHelper.read_images_in_dir(folder)
    return get_ruled_files(files, rules)


def process_folder_group(folder, output, saveStructure=False, rules=default_sample):
    FileHelper.create_or_clear_dir(output)
    for d in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, d)):
            files = process_folder(os.path.join(folder, d), rules)
            if saveStructure:
                copyto = os.path.join(output, d)
                os.makedirs(copyto)
            else:
                copyto = output

            for f in files:
                shutil.copy2(os.path.join(folder, d, f), copyto)


def process_arguments():
    parser = ArgParserWithDefaultHelp(description='Extract frames sample tool from several folders')
    parser.add_argument('folder', help='Folder, that contains folders with frames')
    parser.add_argument('output', help='Folder, to copy selected frames')
    parser.add_argument('-s', '--size', dest='size', default='middle', choices=['middle', 'small', 'extrasmall'], help='Size of sample')
    parser.add_argument('--save-structure', dest='saveStructure', action='store_true', help='Save folder structure in output folder')
    parser.set_defaults(saveStructure=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = process_arguments()
    if args.size == 'extrasmall':
        rules = extra_small_sample
    elif args.size == 'small':
        rules = small_sample
    else:
        rules = middle_sample

    process_folder_group(args.folder, args.output, args.saveStructure, rules)