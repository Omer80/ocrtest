import random


def createNeighbourWindows(image, x, y, amount=7):
    coordinates = set()
    while len(coordinates) < amount:
        nx = int(random.gauss(x, image.shiftSize[0] / 2))
        ny = int(random.gauss(y, image.shiftSize[1] / 2))
        if nx + image.windowSize[0] < image.image.shape[0] and ny + image.windowSize[1] < image.image.shape[1]:
            coordinates.add((nx, ny))

    return coordinates
    # return [image.getWindow(nx, ny) for nx, ny in coordinates]

