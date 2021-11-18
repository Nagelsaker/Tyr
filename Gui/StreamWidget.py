import time
from PyQt5.QtWidgets import  QWidget, QLabel
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QSize, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from FSM import FSM


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    activateGesture = pyqtSignal(int)

    def __init__(self, parent, w, h):
        super().__init__(parent)
        self.w = w
        self.h = h
        self.fsm = FSM()

    def run(self):
        self.fsm.run(self)
    
    def setWristThreshold(self, threshold):
        self.fsm.setWristThreshold(threshold)
    
    def setFingerThreshold(self, threshold):
        self.fsm.setFingerThreshold(threshold)
    
    def setThumbThreshold(self, threshold):
        self.fsm.setThumbThreshold(threshold)

class Stream(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # create a label
        self.label = QLabel(self)
        scale = 1/2
        self.w = int(1920*scale)
        self.h = int(1080*scale)
        self.setMinimumSize(QSize(self.w, self.h))
        # self.label.move(280, 120)
        self.label.resize(self.w, self.h)

        self.th = Thread(self, self.w-5, self.h-5)
        self.th.changePixmap.connect(self.setImage)
        self.th.activateGesture.connect(self.parent().parent().activateGesture)
        self.th.start()

    def setSize(self, width, height):
        self.label.resize(width, height)
        self.w = width
        self.h = height
    
    def close(self):
        self.th.requestInterruption()
        time.sleep(0.2)
        while self.th.isRunning():
            continue

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))