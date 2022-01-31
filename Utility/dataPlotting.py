import csv
import utils
import numpy as np
from constants import *
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection

state = {
    ST_STOP : "STOP",
    ST_MOVE_BACKWARD : "MOVE B",
    ST_MOVE_FORWARD : "MOVE F",
    ST_GRIP : "GRIP",
    ST_UNGRIP : "UNGRIP",
    ST_HEIGHT : "MOVE H",
    ST_TILT_DOWN : "TILT D",
    ST_TILT_UP : "TILT U",
    ST_TURN_LEFT : "TURN L",
    ST_TURN_RIGHT : "TURN R"
}


stateToInt = {
    "STOP" : ST_STOP,
    "MOVE B" : ST_MOVE_BACKWARD,
    "MOVE F" : ST_MOVE_FORWARD,
    "GRIP" : ST_GRIP,
    "UNGRIP" : ST_UNGRIP,
    "MOVE H" : ST_HEIGHT,
    "TILT D" : ST_TILT_DOWN,
    "TILT U" : ST_TILT_UP,
    "TURN L" : ST_TURN_LEFT,
    "TURN R" : ST_TURN_RIGHT
}



def loadRobotPoses(pathToData):
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp, delimiter=",")
        poseData = []

        for line in reader:
            if len(line) == 1 and line[0] == "None":
                poseData.append([[None, None, None], [None, None, None, None]])
                continue

            if len(line) != 9 or line[0] != "position" and line[4] != "orientation":
                raise Exception("Robot poses could not be loaded, format not recognized")

            poseData.append([list(map(float, line[1:4])), list(map(float, line[5:]))])
    return poseData


def loadFSMStates(pathToData):
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp)
        
        stateData = []
        for line in reader:
            if line[0] == "None":
                stateData.append(None)
                continue
            
            stateData.append(state[int(line[0])])
    return stateData


def loadHandPoints(pathToData):
    '''
    HandPoints are on the format:
    idx, X, Y, Z, Visibility, Depth
    '''
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp)

        pointsData = []

        for line in reader:
            if line[0] == "None":
                points.append([None, None, None, None, None])
                continue
            if len(line) % 6 != 0:
                raise Exception("HandPoints could not be loaded, format not recognized")
            points = []
            for i in range(0, len(line), 6):
                points.append(list(map(float, line[i+1:i+6])))
            pointsData.append(points)
    return pointsData


def loadTimeSteps(pathToData):
    with open(f"{pathToData}", "r") as fp:
        reader = csv.reader(fp)
        
        tStepData = []
        for line in reader:
            if line[0] == "None":
                tStepData.append(None)
                continue
            
            tStepData.append(float(line[0]))
    return tStepData

def plotTrajectory3D(X, Y, Z, T):
    n = len(T)
    points = np.array([X, Y, Z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(T.min(), T.max())
    cmap=plt.get_cmap('viridis')
    colors=[cmap(float(i)/(n-1)) for i in range(n-1)]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(n-1):
        segii=segments[i]
        lii,=ax.plot(segii[:,0],segii[:,1],segii[:,2],color=colors[i],linewidth=2)
        lii.set_solid_capstyle('round')

    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    ax.set_xticks(np.arange(min(X), max(X), (max(X)-min(X))/6 ))
    ax.set_yticks(np.arange(min(Y), max(Y), (max(Y)-min(Y))/3 ))
    ax.set_zticks(np.arange(min(Z), max(Z), (max(Z)-min(Z))/2 ))
    ax.set_xlabel("X (m)", labelpad=30)
    ax.set_ylabel("Y (m)", labelpad=10)
    ax.set_zlabel("Z (m)", labelpad=10)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.show()


def plotTrajectory2D(psi, radius, T, prefix):
    points = np.array([np.deg2rad(psi), radius]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # norm = plt.Normalize(states.min(), states.max())
    norm = plt.Normalize(T.min(), T.max())
    lines = LineCollection(segments, cmap='viridis', norm=norm)
    lines.set_array(T)
    # lines.set_linewidth(2)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax.plot(np.deg2rad(psi), radius)
    fig.set_size_inches(10, 10)
    fig.set_dpi(100)
    l = ax.add_collection(lines)
    ax.set_ylabel("Radius (m)", rotation=0, labelpad=-70, size=11)
    ax.set_title("Trajectory")
    ax.set_yticks(np.arange(0, max(radius), 0.05))
    ax.set_xticks(np.arange(np.pi, -np.pi, -np.pi/4))
    ax.set_thetalim(-np.pi, np.pi)
    # ax.set_xticks(np.deg2rad([0, -45, -90, -135, 180, 135, 90, 45]))
    ax.set_rmax(max(radius)+0.01)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar = fig.colorbar(l, ax=ax, )
    cbar.ax.set_ylabel("Time (s)", rotation=-270, labelpad=20)
    # plt.show()
    plt.savefig(f"{prefix}_trajectory.eps")

def plotPoseAndStates(poseData, stateData, timeSteps, prefix):
    N = len(poseData)
    X = np.array([poseData[i][0][0] for i in range(N)])
    Y = np.array([poseData[i][0][1] for i in range(N)])
    Z = np.array([poseData[i][0][2] for i in range(N)])
    T = np.array(timeSteps)
    print(f"\nOperations lasted {T[-1]:.2f} seconds\n")

    quaternions = np.array([
        [poseData[i][1][0] for i in range(N)],
        [poseData[i][1][1] for i in range(N)],
        [poseData[i][1][2] for i in range(N)],
        [poseData[i][1][3] for i in range(N)]
    ]).T

    eulerAnglesRad = utils.euler_from_quaternion(quaternions)
    eulerAngles = np.rad2deg(eulerAnglesRad)

    radius = np.sqrt(X**2 + Y**2)
    psi = np.rad2deg(np.arctan2(Y, X))
    
    plotTrajectory3D(X, Y, Z, T)
    plotTrajectory2D(psi, radius, T, prefix)


    fig, axs = plt.subplots(5,1)
    fig.set_size_inches(10, 16.5)
    fig.set_dpi(200)

    # Plot height
    axs[0].plot(T, Z, label="Z")
    axs[0].legend()
    axs[0].set(ylabel="(m)")
    axs[0].set_title("Height")
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].grid()

    # Plot polar coordinates
    axs[1].plot(T, radius, label="r")
    axs[1].legend()
    axs[1].set(ylabel="Radius (m)")
    axs[1].set_title("Polar Coordinates")
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].grid()

    axs[2].plot(T, psi, label=r"$\beta$")
    axs[2].legend()
    axs[2].set(ylabel="Horizontal rot (Deg)")
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].grid()

    # Euler angles of end-effector
    axs[3].plot(T, eulerAngles, label=[r"$\psi$", r"$\theta$", r"$\phi$"])
    axs[3].legend()
    axs[3].set(ylabel="Euler angles (Deg)")
    axs[3].set_title("Orientation")
    axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[3].grid()

    # Plotting State
    axs[4].step(T, stateData, label="State")
    axs[4].legend()
    axs[4].set(ylabel="States")
    axs[4].set(xlabel="Time (s)")
    axs[4].set_title("State")
    axs[4].grid()

    # Plotting State
    # axs[5].step(T, np.rad2deg(eulerAnglesRad[:-5,2] - np.arctan2(Y, X)[5:]), label="State")
    # axs[5].legend()
    # axs[5].set(ylabel="States")
    # axs[5].set(xlabel="Time (s)")
    # axs[5].set_title("State")
    # axs[5].grid()
    
    plt.show()
    # plt.savefig(f"{prefix}_Pose_and_States.eps")

if __name__ == "__main__":
    # Load data
    # prefix = "log/3_2021_12_09_"
    prefix = "log/11_2021_12_10_" # 15 seconds for 11
    poseData = loadRobotPoses(f"{prefix}RobotPoseLogger.csv")
    stateData = loadFSMStates(f"{prefix}FSMStateLogger.csv")
    # pointsData = loadHandPoints(f"{prefix}HandPointsLogger.csv")
    timeSteps = loadTimeSteps(f"{prefix}TimeStepLogger.csv")

    # Plotting Pose:
    plotPoseAndStates(poseData, stateData, timeSteps, prefix)