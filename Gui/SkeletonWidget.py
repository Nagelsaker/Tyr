import time
from PyQt5.QtWidgets import  QWidget, QLabel
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QSize, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import numpy as np


class SkeletonWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # create a label
        self.label = QLabel(self)
        self.w = int(300)
        self.h = int(540)
        self.setMinimumSize(QSize(self.w, self.h))
        # self.label.move(280, 120)
        self.label.resize(self.w, self.h)
        test = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1]])
        self.updatePoints(test)

    def updatePoints(self, landmarks):
        pass
