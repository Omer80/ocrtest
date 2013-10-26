
from image.processing import Image
from classifier.test_images import process_image
from misc.file_helper import FileHelper


class HashtagRecognition(object):
    def __init__(self, classifier):
        self.classifier = classifier

    def process_image(self, image):
        """
        Process single image.processing.Image
        Returns: boolean
            True, if image has hashtag inside
            False, otherwise
        """
        return process_image(self.classifier, image)

    def process_file(self, filename):
        """
        Process single image file
        Returns: boolean
            True, if image has hashtag inside
            False, otherwise
        """
        return self.process_image(Image(filename))

    def process_file_list(self, filenames):
        """
        Process files group
        Returns: (positive, negative)
            Tuple with two lists: positive and negative file names
        """
        # todo: parallel processing
        # todo: use information about two neighbours frames (if hashtag on the same position in two neighbour frames - it is good sign)
        positive, negative = [], []
        for f in filenames:
            if self.process_file(f):
                positive.append(f)
            else:
                negative.append(f)

        return positive, negative

    def process_folder(self, folder):
        """
        Process folder
        Returns: (positive, negative)
            Tuple with two lists: positive and negative file names
        """
        return self.process_file_list(FileHelper.read_images_in_dir(folder))