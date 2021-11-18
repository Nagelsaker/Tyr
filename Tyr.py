import sys
from PyQt5.QtWidgets import QApplication
from Gui.OperatorPanel import MainWindow
import qdarkstyle


if __name__ == "__main__":
    app = QApplication(sys.argv)

    darkStylesheet = qdarkstyle.load_stylesheet_pyqt()
    app.setStyleSheet(darkStylesheet)
    
    window = MainWindow()
    window.show()
    app.exec()