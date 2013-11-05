import os

from image.pattern_matcher import PatternMatcher
from misc.file_helper import FileHelper

if __name__ == '__main__':
    import sys

    # for dir, filename in FileHelper.read_images_in_dir_recursively('frames_tagged_291013_goodbad'):
    #     pm = PatternMatcher()
    #     pattern_dir = 'frames_tagged_291013_goodbad_ht/' + '/'.join((dir.split('/')[1:]))
    #     bfn, ext = os.path.splitext(filename)
    #
    #     pm.load_patterns([os.path.join(pattern_dir, bfn+'_ht'+ext)], convert2GS=True)
    #     pm.match(os.path.join(dir, filename), generateHist=True)

    pm = PatternMatcher()
    # pm.load_patterns([sys.argv[1]], convert2GS=True)
    pm.load_patterns_folder(sys.argv[1])
    if os.path.isdir(sys.argv[2]):
        for dir, filename in FileHelper.read_images_in_dir_recursively(sys.argv[2]):
            print os.path.join(dir, filename)
            pm.match(os.path.join(dir, filename), visualise=True)
    else:
        pm.match(sys.argv[2], visualise=True)