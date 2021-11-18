from PyQt5.QtWidgets import  QWidget, QLabel
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPixmap


class GestureImage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.label = QLabel(self)
        self.w = 230 # px
        self.h = 230 # px
        self.setMinimumSize(QSize(self.w, self.h))
        self.label.resize(self.w, self.h)
        self.image = None
        self.activeImage = None
    
    def setImage(self, imagePath):
        qimage = QImage(imagePath).scaled(self.w-10, self.h-10)
        self.image = QPixmap.fromImage(qimage)
        self.label.setPixmap(self.image)
    
    def setActiveImage(self, imagePath):
        qimage = QImage(imagePath).scaled(self.w-10, self.h-10)
        self.activeImage = QPixmap.fromImage(qimage)

    def activate(self):
        self.label.setPixmap(self.activeImage)

    def deactivate(self):
        self.label.setPixmap(self.image)