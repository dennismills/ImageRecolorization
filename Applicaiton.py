import sys
from PyQt5.QtWidgets import QApplication


class Application(QApplication):
    def __init__(self):
        super().__init__(sys.argv)

    def start(self):
        sys.exit(self.exec_())
