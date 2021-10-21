from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, \
                            QAction, QFileDialog, QErrorMessage, \
                            QTextEdit, QInputDialog, QLineEdit, \
                            QWidget, QPushButton, QLabel

import os
from ImageManager import Image
from PyQt5.QtGui import QResizeEvent

class Window (QMainWindow):
    def __init__(self, title, width, height, imageManager):
        super().__init__()
        self.title = title
        self.width = width
        self.height = height

        self.desktop = QDesktopWidget().screenGeometry()

        self.setGeometry(int((self.desktop.width() / 2) - (self.width / 2)),
                         int((self.desktop.height() / 2) - (self.height / 2)),
                         self.width,
                         self.height)

        self.setWindowTitle(self.title)
        self.statusBar()

        self.statusBar()

        self.log = QTextEdit(self)
        self.logScaleY = 0.25
        self.previewScale = 0.45
        self.log.setGeometry(0, self.height - (self.height * self.logScaleY), self.width, self.height * self.logScaleY)
        self.log.show()
        self.log.setReadOnly(True)
        self.logText = ""

        self.previewPane = QWidget(self)
        self.previewPane.setGeometry(self.width * 0.025, self.height * 0.1,
                                     self.width * self.previewScale, self.height * self.previewScale)
        self.previewPane.setStyleSheet("background: #777777;")
        self.nextButton = QPushButton(self)
        self.lastButton = QPushButton(self)
        self.previewImage = QLabel(self)
        self.coloredImageCounter = 0
        self.nextButton.setText("------>")
        self.lastButton.setText("<------")

        nextButtonX = self.previewPane.geometry().x() + self.previewPane.geometry().width() - \
                  self.previewPane.geometry().width() * 0.20

        nextButtonY = self.previewPane.geometry().y() + self.previewPane.geometry().height() + \
                  (self.previewPane.geometry().height() * 0.05)

        lastButtonX = self.previewPane.geometry().x()

        lastButtonY = self.previewPane.geometry().y() + self.previewPane.geometry().height() + \
                      (self.previewPane.geometry().height() * 0.05)

        self.nextButton.setGeometry(nextButtonX, nextButtonY, 75, 25)
        self.lastButton.setGeometry(lastButtonX, lastButtonY, 75, 25)
        self.nextButton.show()
        self.lastButton.show()

        self.nextButton.clicked.connect(lambda: self.nextImage(imageManager))
        self.lastButton.clicked.connect(lambda: self.lastImage(imageManager))

        bar = self.menuBar()
        bar.setNativeMenuBar(False)
        fileMenu = bar.addMenu('&File')
        loadAction = QAction("Load", self)
        loadAction.triggered.connect(lambda _: self.__load(imageManager))
        fileMenu.addAction(loadAction)

        self.show()

    def nextImage(self, imageManager):
        self.coloredImageCounter += 1
        if len(imageManager.images) > 0:
            imageManager.images[self.coloredImageCounter].resize((self.previewPane.geometry().width(),
                                                                  self.previewPane.geometry().height()))
            self.previewImage.setPixmap(imageManager.images[self.coloredImageCounter].toPixmap())

    def lastImage(self, imageManager):
        self.coloredImageCounter -= 1
        if len(imageManager.images) > 0:
            imageManager.images[self.coloredImageCounter].resize((self.previewPane.geometry().width(),
                                                                  self.previewPane.geometry().height()))
            self.previewImage.setPixmap(imageManager.images[self.coloredImageCounter].toPixmap())

    def pushToLog(self, data, color = "FFFFFF"):
        spanText = "<span style =\"color:#" + color + ";\">"
        self.logText += spanText
        self.logText += data
        spanEndText = "</span>"
        self.logText += spanEndText + "<br>"

    def publishLog(self):
        self.log.setText(self.logText)

    def fileWalk(self, folder, typeFilter = []):
        output = []
        for root, directoryNames, fileNames in os.walk(folder):
            for file in fileNames:
                if typeFilter is not None and len(typeFilter) > 0:
                    extension = file[file.find("."): ]
                    #print("Is {} in typeFilter: {}".format(extension, extension in typeFilter))
                    if extension in typeFilter:
                        output.append(file)
                else:
                    output.append(file)

        return output

    def __load(self, imageManager):
        folder = QFileDialog.getExistingDirectory(self, "Select a directory to load images from")
        if not len(folder) > 0:
            msgBox = QErrorMessage()
            msgBox.showMessage("No folder was selected")
            msgBox.exec()
        else:
            classLabel, ok = QInputDialog.getText(self, "Class Label?", "Class Label", QLineEdit.Normal)

            if len(classLabel) <= 0:
                classLabel = None
            baseFiles = os.listdir(folder)
            supportedFileTypes = [".jpg", ".png", ".bmp", ".tiff"]
            files = []
            for file in baseFiles:
                if os.path.isdir(folder + "/" + file):
                    subFiles = self.fileWalk(folder + "/" + file, supportedFileTypes)
                    for i in subFiles:
                        files.append(i)

            for file in baseFiles:
                if not os.path.isdir(file):
                    files.append(file)

            for file in files:
                extension = file[file.rfind("."):] # Gets everything from the '.' to the end of the string
                if extension.find(".") >= 0: # It's possible there wasn't a '.', so check for one
                    if extension in supportedFileTypes: # If the extension is in our list of supported ones
                        image = Image(folder + "/" + file, classLabel)
                        self.pushToLog("Attempting to load: {}".format(folder + "/" + file), color = "FFFF00")
                        self.pushToLog("Image Info : {}".format(image.getString()), color = "00FFFF")
                        imageManager.add(image)
                    else: # If the extension doesn't match supported types
                        #TODO: Maybe nothing? Skip non supported types, but maybe pass a warning somewhere
                        self.pushToLog("Couldn't load: {}".format(folder + "/" + file), color = "FF0000")

            self.pushToLog("Loading Complete", color = "00FF00")
            self.publishLog()
            self.previewImage = QLabel(self.previewPane)
            self.previewImage.setGeometry(0,
                                          0,
                                          self.previewPane.geometry().width(),
                                          self.previewPane.geometry().height()
                                          )

            imageManager.images[self.coloredImageCounter].resize((self.previewPane.geometry().width(),
                                                                  self.previewPane.geometry().height()))

            self.previewImage.setPixmap(imageManager.images[self.coloredImageCounter].toPixmap())
            self.previewImage.show()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.width = self.geometry().width()
        self.height = self.geometry().height()
        self.log.setGeometry(0, (self.height - self.logScaleY * self.height), self.width, self.height * self.logScaleY)

        self.previewPane.setGeometry(self.width * 0.025, self.height * 0.1,
                                     self.width * self.previewScale, self.height * self.previewScale)


