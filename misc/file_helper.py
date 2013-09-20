# coding=utf-8
import shutil

__author__ = 'Roman Podlinov'

import os

class FileHelper:

    @staticmethod
    def remove_files_in_dir(path):
        if (path == '/' or path == "\\"):
            return
        else:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    shutil.rmtree(os.path.join(root, name))

    @staticmethod
    def read_images_in_dir(dir, load_pattern='.png:.jpg:.jpeg:.gif'):
        load_pattern = load_pattern.split(':')
        matches = []
        for root, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                # if filename.endswith(('.jpg', '.jpeg', '.gif', '.png')):
                if filename.endswith(tuple(load_pattern)):
                    matches.append(filename)

        # return sorted(matches, key=lambda item: (int(item.partition(' ')[0])
        #                            if item[0].isdigit() else float('inf'), item))

        return matches

    @staticmethod
    def create_or_clear_dir(dir):
        if os.path.exists(dir):
            FileHelper.remove_files_in_dir(dir)
        else:
            os.makedirs(dir)
