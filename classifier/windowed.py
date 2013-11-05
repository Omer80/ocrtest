from image.processing import create_image


class WindowedFeatureClassifier(object):
    def __init__(self, classifier, imageFactory=create_image):
        self.classifier = classifier
        self.imageFactory = imageFactory

    def process_image(self, image, includeNegativeWindows=False):
        _, windows = image.process()
        predicted = self.classifier.predict(windows)

        result = []
        for i, (w, p) in enumerate(zip(windows, predicted)):
            xc = (i / image.windowsAmountInfo[1])
            yc = (i % image.windowsAmountInfo[1])
            x = xc * image.shiftSize[0]
            y = yc * image.shiftSize[1]

            if p:
                additionalWindows = image.createNeighbourWindows(x, y, certainThatWithTag=False)
                featured = [image.process_window(window) for window in additionalWindows]
                additionalPredicted = self.classifier.predict(featured)
                r = (sum(additionalPredicted) + 1) / float(len(additionalPredicted) + 1)
            else:
                r = 0

            if includeNegativeWindows or p:
                x += image.bounds[0].start-1
                y += image.bounds[1].start-1
                result.append((r, (x, y, x+image.windowSize[0], y+image.windowSize[1])))

        return image.imagePath, result

    def process_file(self, filename):
        return self.process_image(self.imageFactory(filename))