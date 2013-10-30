import os
import shutil

import cv2

from misc.file_helper import FileHelper
from ocr_utils import ArgParserWithDefaultHelp


class FolderTagger(object):
    def __init__(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.create_tag)

        self.drawing = False
        self.image = None
        self.showedImage = None
        self.ix, self.iy = 0, 0

        self.tags = []
        self.currentTag = None
        self.isNewTag = False

        self.history = []

    def move(self, fromPath, toPath, history=True):
        print "Move '%s' to '%s'" % (fromPath, toPath)
        if not os.path.exists(toPath):
            os.makedirs(toPath)
        shutil.move(fromPath, toPath)
        if history:
            self.history.insert(0, (fromPath, toPath))

    def wait_for_action(self):
        stopActions = {
            27: 'break',
            32: 'next',
        }
        showTagAction = set(range(48, 58))
        moveActions = set([97, 98, 102, 103, 110, 115, 116])

        key = 0
        while key not in stopActions:
            cv2.imshow('image', self.showedImage)
            key = cv2.waitKey(20) & 0xFF
            if key in showTagAction:
                self.isNewTag = False
                if key == 48:
                    self.clear()
                    for tag in self.tags:
                        self.show_tag(tag, clear=False)
                else:
                    self.clear()
                    try:
                        self.show_tag(self.tags[key - 49])
                    except IndexError:
                        pass

            if key in moveActions:
                # n 110
                # g 103
                # b 98
                # t 116
                # f 102
                taggedFolder = None
                if self.currentTag:
                    t = self.currentTag
                    taggedFolder = '%s_%dx%dx%dx%d' % (self.uplevelFolder, t[0], t[1], t[2], t[3])
                if key == 110:
                    self.move(self.filename, os.path.join(self.negativePath, self.uplevelFolder))
                    return 'next'
                elif taggedFolder:
                    if self.isNewTag:
                        for tag in self.tags:
                            if tag == self.currentTag:
                                self.isNewTag = False
                                break
                        if self.isNewTag:
                            self.tags.insert(0, self.currentTag)
                            if len(self.tags) > 9:
                                self.tags = self.tags[:9]
                            self.isNewTag = False

                    if key == 103:
                        self.move(self.filename, os.path.join(self.goodPath, taggedFolder))
                    if key == 98:
                        self.move(self.filename, os.path.join(self.badPath, taggedFolder))
                    if key == 116:
                        self.move(self.filename, os.path.join(self.twitterPath, taggedFolder))
                    if key == 102:
                        self.move(self.filename, os.path.join(self.facebookPath, taggedFolder))
                    if key == 97:   # a
                        self.move(self.filename, os.path.join(self.awfulPath, taggedFolder))
                    if key == 115:  # s
                        self.move(self.filename, os.path.join(self.atPath, taggedFolder))

                    return 'next'

            if key == 8 and self.history:
                # undo last action
                last = self.history.pop(0)
                if len(last) == 2:
                    toPath, fn = os.path.split(last[0])
                    self.move(os.path.join(last[1], fn), toPath, history=False)
                self.filelist.insert(0, self.filename)
                self.filelist.insert(0, last[0])
                return 'next'

            # if key != 255:
            #     print key

        if stopActions[key] == 'next':
            self.history.insert(0, (self.filename, ))

        return stopActions[key]

    def clear(self):
        self.showedImage = self.image.copy()
        self.currentTag = None
        self.isNewTag = False

    def show_tag(self, tag, clear=True):
        if clear:
            self.clear()
        cv2.rectangle(self.showedImage, (tag[0], tag[1]), (tag[2], tag[3]), (255, 255, 255))
        self.currentTag = tag

    def create_tag(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.showedImage is not None:
                self.show_tag((self.ix, self.iy, x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                if self.showedImage is not None:
                    tag = (min(self.ix, x), min(self.iy, y), max(self.ix, x), max(self.iy, y))
                    self.show_tag(tag)
                    self.currentTag = tag
                    self.isNewTag = True

    def load(self, filename):
        self.image = cv2.imread(filename)
        self.showedImage = self.image.copy()

    def show_next(self):
        if not self.filelist:
            return False

        self.filename = self.filelist.pop(0)
        self.load(self.filename)
        cv2.imshow('image', self.showedImage)
        if self.currentTag is not None:
            self.show_tag(self.currentTag, False)

        return True

    def tag_folder(self, folder, output):
        print folder
        if folder[-1] == '\\':
            folder = folder[:-1]
        self.currentFolder = folder
        self.uplevelFolder = os.path.split(folder)[-1]

        self.negativePath = os.path.join(output, 'negative')
        self.goodPath = os.path.join(output, 'good')
        self.badPath = os.path.join(output, 'bad')
        self.awfulPath = os.path.join(output, 'awful')
        self.twitterPath = os.path.join(output, 'twitter')
        self.facebookPath = os.path.join(output, 'facebook')
        self.atPath = os.path.join(output, 'at')

        self.filelist = sorted(FileHelper.read_images_in_dir(folder, includeDir=True))
        if not self.show_next():
            shutil.rmtree(folder)
            return True
        while True:
            action = self.wait_for_action()
            if action == 'break':
                return False
            elif action == 'next':
                if not self.show_next():
                    shutil.rmtree(folder)
                    return True

    def destroy_windows(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    '''
    $ find /share/strg/frames/ -mtime -10 -type d >temp_ls
    remove first line from file
    $ while read file;do echo ln -s "$file" ./frames_291013/; done <temp_ls    <-- remove echo
    '''
    parser = ArgParserWithDefaultHelp(description='Tagging images tool')
    parser.add_argument('folder', help='Folder, that contains folders with frames')
    parser.add_argument('output', help='Output folder  for tagged images')
    args = parser.parse_args()

    ft = FolderTagger()
    for fn in os.listdir(args.folder):
        if os.path.isdir(os.path.join(args.folder, fn)):
            if not ft.tag_folder(os.path.join(args.folder, fn), args.output):
                break