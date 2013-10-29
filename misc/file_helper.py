# coding=utf-8
__author__ = 'Roman Podlinov'

import os
import shutil


class FileHelper:

    @staticmethod
    def remove_files_in_dir(path):
        if path == '/' or path == "\\":
            return

        for obj in os.listdir(path):
            if os.path.isfile(obj):
                os.remove(os.path.join(path, obj))
            else:
                shutil.rmtree(os.path.join(path, obj))

        # for root, dirs, files in os.walk(path, topdown=False):
        #     for name in files:
        #         os.remove(os.path.join(root, name))
        #     for name in dirs:
        #         shutil.rmtree(os.path.join(root, name))

    @staticmethod
    def read_images_in_dir(dir, load_pattern='.png:.jpg:.jpeg:.gif', includeDir=False):
        load_pattern = tuple([ext.lower() for ext in load_pattern.split(':')])
        matches = []
        for filename in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, filename)) and filename.lower().endswith(load_pattern):
                if includeDir:
                    matches.append(os.path.join(dir, filename))
                else:
                    matches.append(filename)

        return matches

    @staticmethod
    def create_or_clear_dir(dir):
        if os.path.exists(dir):
            FileHelper.remove_files_in_dir(dir)
        else:
            os.makedirs(dir)
