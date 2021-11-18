from PyQt5 import QtCore, QtGui, QtWidgets
from Gui.Ui_SettingsDialog import Ui_Dialog



class SettingsDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)