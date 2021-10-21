import cv2
from PyQt5.QtGui import QImage, QPixmap


class Image:
    def __init__(self, fileName, label = None):
        self.fileName = fileName
        self.imageData = cv2.imread(fileName)
        self.pixels = self.imageData.data
        cv2.cvtColor(self.imageData, cv2.COLOR_BGR2RGB, self.imageData)
        self.width, self.height, self.channels = None, None, None
        if self.imageData is not None:
            self.width, self.height, self.channels = self.imageData.shape

        self.label = label
        self.stride = self.imageData.strides

    def copy(i):
        pass # TODO Make copy of image

    def getImageData(self):
        return self.imageData

    def getSize(self):
        return self.width, self.height

    def getChannels(self):
        return self.channels

    def getFileName(self):
        return self.fileName

    def getLabel(self):
        return self.label

    def getStride(self):
        return self.stride

    def getString(self):
        return "{}: Size: ({}) Channels: {} Class: {}".format(self.getFileName(),
                                                                    self.getSize(),
                                                                    self.getChannels(),
                                                                    self.getLabel())

    def resize(self, newDim):
        self.imageData = cv2.resize(self.imageData, newDim, interpolation = cv2.INTER_CUBIC)
        self.width, self.height, self.channels = self.imageData.shape
        self.pixels = self.imageData.data

    def toPixmap(self):
        return QPixmap.fromImage(QImage(self.pixels, self.height, self.width, QImage.Format_RGB888))

class ImageManager:
    def __init__(self):
        self.images = []

    def add(self, image):
        self.images.append(image)
