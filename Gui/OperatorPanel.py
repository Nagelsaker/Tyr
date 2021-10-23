from PyQt5.QtWidgets import  QMainWindow
from Gui.MainWindow import Ui_MainWindow
from Hand.HandModel import *


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.show()

    def closeEvent(self, event):
        self.videoStream.close()
        event.accept()