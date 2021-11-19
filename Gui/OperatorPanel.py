from PyQt5.QtWidgets import  QMainWindow
from Gui.Ui_MainWindow import Ui_MainWindow
from Gui.SettingsDialog import SettingsDialog
from Hand.HandModel import *
import json

STOP = 0
GRIP = 1
UNGRIP = 2
PRECISION = 3
TILT_UP = 4
TILT_DOWN = 5
MOVE_HEIGHT = 6



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.gestureImages = [] # Wrapper
        self.setupUi(self)

        # Setup gesture images
        self.gestureImage0.setImage("data/gesture0.png")
        self.gestureImage0.setActiveImage("data/gesture0_active.png")
        self.gestureImages.append(self.gestureImage0)
        self.gestureImage1.setImage("data/gesture1.png")
        self.gestureImage1.setActiveImage("data/gesture1_active.png")
        self.gestureImages.append(self.gestureImage1)
        self.gestureImage2.setImage("data/gesture2.png")
        self.gestureImage2.setActiveImage("data/gesture2_active.png")
        self.gestureImages.append(self.gestureImage2)
        self.gestureImage3.setImage("data/gesture3.png")
        self.gestureImage3.setActiveImage("data/gesture3_active.png")
        self.gestureImages.append(self.gestureImage3)
        self.gestureImage4.setImage("data/gesture4.png")
        self.gestureImage4.setActiveImage("data/gesture4_active.png")
        self.gestureImages.append(self.gestureImage4)
        self.gestureImage5.setImage("data/gesture5.png")
        self.gestureImage5.setActiveImage("data/gesture5_active.png")
        self.gestureImages.append(self.gestureImage5)
        self.gestureImage6.setImage("data/gesture6.png")
        self.gestureImage6.setActiveImage("data/gesture6_active.png")
        self.gestureImages.append(self.gestureImage6)
        self.currentGesture = -1

        self.addAction(self.actionPreferences)
        self.actionPreferences.triggered.connect(self.openDialog)

        # Threshold spin boxes
        self.threshold_wristUp.valueChanged.connect(self.setWristThreshold)
        self.threshold_wristDown.valueChanged.connect(self.setWristThreshold)
        self.threshold_fingerAng1.valueChanged.connect(self.setFingerThreshold)
        self.threshold_thumbAng1.valueChanged.connect(self.setThumbThreshold)
        self.threshold_thumbAng2.valueChanged.connect(self.setThumbThreshold)

        # Buttons
        self.saveData.clicked.connect(self.saveDataPoints)

        # Load default threshold values
        f = open("settings.json")
        settings = json.load(f)
        self.threshold_wristUp.setValue(settings["wristAngle_threshold"][0])
        self.threshold_wristDown.setValue(settings["wristAngle_threshold"][1])
        self.threshold_fingerAng1.setValue(settings["fingerAngle_threshold"])
        self.threshold_thumbAng1.setValue(settings["thumbAngle_threshold"][0])
        self.threshold_thumbAng2.setValue(settings["thumbAngle_threshold"][1])
        
    def setWristThreshold(self):
        th1 = self.threshold_wristUp.value()
        th2 = self.threshold_wristDown.value()
        self.videoStream.th.setWristThreshold([th1, th2])
        
    def setFingerThreshold(self):
        th1 = self.threshold_fingerAng1.value()
        self.videoStream.th.setFingerThreshold(th1)
        
    def setThumbThreshold(self):
        th1 = self.threshold_thumbAng1.value()
        th2 = self.threshold_thumbAng2.value()
        self.videoStream.th.setThumbThreshold([th1, th2])

    def setDepthValue(self, value):
        self.depthDisplay.display(value)

    def openDialog(self):
        settingsDialog = SettingsDialog(self)
        settingsDialog.show()

    def activateGesture(self, idx):
        if idx != self.currentGesture:
            for i in range(len(self.gestureImages)):
                if i == idx:
                    self.currentGesture = idx
                    self.gestureImages[i].activate()
                else:
                    self.gestureImages[i].deactivate()
        if idx == -1: self.currentGesture = idx
    
    def updateSkeleton(self, landmarks):
        self.skeletonWidget.updatePoints(landmarks)
        # self.skeletonWidget_synthetic.updatePoints(landmarks)
    
    def saveDataPoints(self):
        self.skeletonWidget.save()
        self.videoStream.saveImage()


    def closeEvent(self, event):
        self.videoStream.close()
        event.accept()