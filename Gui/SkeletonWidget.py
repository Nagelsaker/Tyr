from PyQt5.QtWidgets import  QWidget, QVBoxLayout
from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from Utility.utils import generateFilename
import numpy as np
import json

class SkeletonWidget(QWidget):
    angle1 = 0
    angle2 = 15
    dir = 1
    rotate = True
    landmarks = None

    def __init__(self, parent=None):
        super().__init__(parent)

        f = open("settings.json")
        settings = json.load(f)
        self.useDepth = settings["useDepth"] == 1
        self.pathToDataset = settings["pathToDataset"]
        self.w = int(650)
        self.h = int(500)
        self.setMaximumSize(QSize(self.w, self.h))

        # a figure instance to plot on
        self.figure = Figure(figsize=(self.w, self.h))
        self.ax = self.figure.add_subplot(projection="3d")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.ax.set_zticklabels([])
        self.ax.set_facecolor("none")

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.figure.set_facecolor("none")
        self.canvas.setStyleSheet("background-color:transparent;")

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        
    def updatePoints(self, landmarks):
        if len(landmarks) == 0: return

        self.landmarks = landmarks

        # print(landmarks)
        X = [d["X"] for d in landmarks]
        Y = [d["Y"] for d in landmarks]
        if self.useDepth:
            Z = [d["Depth"] for d in landmarks]
        else:
            Z = [d["Z"] for d in landmarks]

        n = len(X)
        data = np.array([X, Y, Z, np.ones(n)])
        H = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        data = (data.T @ H).T
        X, Y, Z = data[:3] / data[3]

        # discards the old graph
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.ax.set_zticklabels([])
        self.ax.set_facecolor("none")

        # plot data
        self.ax.scatter(X, Y, Z, marker="o", s=2, c="blue")

        # Thumb
        self.ax.plot([X[0], X[1]], [Y[0], Y[1]], [Z[0], Z[1]], c="red")
        self.ax.plot([X[1], X[2]], [Y[1], Y[2]], [Z[1], Z[2]], c="red")
        self.ax.plot([X[2], X[3]], [Y[2], Y[3]], [Z[2], Z[3]], c="red")
        self.ax.plot([X[3], X[4]], [Y[3], Y[4]], [Z[3], Z[4]], c="red")
        # Index
        self.ax.plot([X[0], X[5]], [Y[0], Y[5]], [Z[0], Z[5]], c="red")
        self.ax.plot([X[5], X[6]], [Y[5], Y[6]], [Z[5], Z[6]], c="red")
        self.ax.plot([X[6], X[7]], [Y[6], Y[7]], [Z[6], Z[7]], c="red")
        self.ax.plot([X[7], X[8]], [Y[7], Y[8]], [Z[7], Z[8]], c="red")
        # Middle finger
        self.ax.plot([X[5], X[9]], [Y[5], Y[9]], [Z[5], Z[9]], c="red")
        self.ax.plot([X[9], X[10]], [Y[9], Y[10]], [Z[9], Z[10]], c="red")
        self.ax.plot([X[10], X[11]], [Y[10], Y[11]], [Z[10], Z[11]], c="red")
        self.ax.plot([X[11], X[12]], [Y[11], Y[12]], [Z[11], Z[12]], c="red")
        # Ring finger
        self.ax.plot([X[9], X[13]], [Y[9], Y[13]], [Z[9], Z[13]], c="red")
        self.ax.plot([X[13], X[14]], [Y[13], Y[14]], [Z[13], Z[14]], c="red")
        self.ax.plot([X[14], X[15]], [Y[14], Y[15]], [Z[14], Z[15]], c="red")
        self.ax.plot([X[15], X[16]], [Y[15], Y[16]], [Z[15], Z[16]], c="red")
        # Pinky finger
        self.ax.plot([X[13], X[17]], [Y[13], Y[17]], [Z[13], Z[17]], c="red")
        self.ax.plot([X[17], X[18]], [Y[17], Y[18]], [Z[17], Z[18]], c="red")
        self.ax.plot([X[18], X[19]], [Y[18], Y[19]], [Z[18], Z[19]], c="red")
        self.ax.plot([X[19], X[20]], [Y[19], Y[20]], [Z[19], Z[20]], c="red")
        self.ax.plot([X[0], X[17]], [Y[0], Y[17]], [Z[0], Z[17]], c="red")

        # Zoom out
        self.ax.xaxis.zoom(-2)
        self.ax.yaxis.zoom(-2)
        self.ax.zaxis.zoom(-2)

        # Rotate automatically
        if self.rotate:
            if self.angle1 == 360: self.angle1 = 0
            if self.angle2 == 45: self.dir = -1
            elif self.angle2 == 15: self.dir = 1
            self.ax.view_init(self.angle2, self.angle1)
            self.angle1 += 1
            self.angle2 += self.dir

        # refresh canvas
        self.canvas.draw()
    
    def save(self):
        skeletonFname = generateFilename(self.pathToDataset, "svg", "skeleton")
        self.figure.savefig(skeletonFname)

        lmFname = generateFilename(self.pathToDataset, "json", "landmarks")
        with open(lmFname, "w") as fout:
            json.dump(self.landmarks, fout)
