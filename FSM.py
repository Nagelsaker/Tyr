from Controller import Controller, Obstacle
import time
import numpy as np
from HandTracking import HandTracking
from HandModel import HandModel

def fsm():
    depthRange = [0.28, 0.42]
    pathTime = 0.2
    imgWidth = 1920
    imgHeight = 1080

    # Obstacles
    floor = Obstacle(zRange=[-99, 0.037])
    ceiling = Obstacle(zRange=[0.456, 99])
    innerCylinder = Obstacle(radiusRange=[0, 0.06])
    outerCylinder = Obstacle(radiusRange=[0.386, 99])
    motor = Obstacle(xRange=[0.175, 99], yRange=[-0.12, 0.05], zRange=[-99, 0.095])
    obstacles = np.array([floor, ceiling, innerCylinder, outerCylinder, motor])

    handTracker = HandTracking()
    hm = HandModel(type="left")
    controller = Controller(imgWidth, imgHeight, depthRange, obstacles=obstacles, pathTime=pathTime)

    standardPose = controller.getPose()
    standardPose["position"]["x"] = 0.148
    standardPose["position"]["y"] = 0.0
    standardPose["position"]["z"] = 0.243
    
    handTracker.startStream()

    try:
        while True: # Tracking loop
            handPoints, image = handTracker.getLiveLandamarks(visualize=True)
            hm.addMeasurement(handPoints)

            controller.updateRobotPose()

            currentGesture = hm.getCurrentGesture()
            palm = hm.getPalmLocation()
            depth, var = hm.getHandDepth()
            if depth == 0 and var == 0:
                continue
            
            pos = controller.getPose()["position"]
            print(f"Current gesture:\t{currentGesture}\tPosition:\t{pos}\r", end="")

            if currentGesture == 0: # Stop moving
                # TODO?
                pass
            elif currentGesture == 1: # Grip
                pass
            elif currentGesture == 2: # Release grip
                pass
            elif currentGesture == 3: # Turn in the xy plane
                controller.incrementXY(palm)
                pass
            elif currentGesture == 4: # Move in rz plane
                controller.incrementRadiusAndZ(palm, depth)
                pass
            elif currentGesture == 5: # Move to standard pose
                controller.requestPose(standardPose, time=0.8)
                # time.sleep(3)

    except KeyboardInterrupt:
        print("\nExiting..")
        handTracker.endStream() # Remember to end the stream
        controller.endController()

#     STATES:
#         q = 0:
#             (No Movement)
#             * Tell the controller to stop moving the manipulator
#             * Jumps to this state immediately when the stop gesture is detected,
#                 or jumps after x seconds of invalid gesture. Remains in previous
#                 state in those x seconds. May not be necessary.
#                 TODO: Figure this out
        
#         q = 1:
#             (Standard Pose)
#             * Controllerm moves manipulator into pre-determined standard pose
        
#         q = 2:
#             (Grip)
#             * Controller asks manipulator to start gripping
#             * Ends when a large enough force is detected, or
#                 when operator stops action
        
#         q = 3:
#             (Ungrip)
#             * Controller asks manipulator to open up its gripper
#             * Ends when gripper is fully opened or, operator
#                 stops action
        
#         q = 4:
#             (Move in the XZ plane)
#             * Controller moves the manipulator in the XZ plane, according to
#                 the position of the operators hand in front of the camera
#             * Will not allow operator to move end-effector in forbidden areas
#             * Stops when correct hand gesture is no longer detected.
        
#         q = 5:
#             (Turn in the XY plane)
#             * Keeps the radius of the circle constant, but lets the operator
#                 turn the manipulator to the right/left according to the position
#                 of the operators hand in front of the camera.
#             * TODO: Test moving up and down as well. Movement constrained to a
#                 cylinder instead of a circle
#             * Will not allow operator to move end-effector in forbidden areas




    
#     Update Operator panel:
#     * Detected hand gesture
#     * 

if __name__ == "__main__":
    fsm()