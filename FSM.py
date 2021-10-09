'''
Setup:
* Start Hand Tracking
* Initiate Hand Model
* Return manipulator to initial position
* Launch Operator as a seperate Panel

While Loop:
    Normal Operations:
    * Get Hand Tracking Info
    * Update Hand Model

    STATES:
        q = 0:
            (No Movement)
            * Tell the controller to stop moving the manipulator
            * Jumps to this state immediately when the stop gesture is detected,
                or jumps after x seconds of invalid gesture. Remains in previous
                state in those x seconds. May not be necessary.
                TODO: Figure this out
        
        q = 1:
            (Standard Pose)
            * Controllerm moves manipulator into pre-determined standard pose
        
        q = 2:
            (Grip)
            * Controller asks manipulator to start gripping
            * Ends when a large enough force is detected, or
                when operator stops action
        
        q = 3:
            (Ungrip)
            * Controller asks manipulator to open up its gripper
            * Ends when gripper is fully opened or, operator
                stops action
        
        q = 4:
            (Move in the XZ plane)
            * Controller moves the manipulator in the XZ plane, according to
                the position of the operators hand in front of the camera
            * Will not allow operator to move end-effector in forbidden areas
            * Stops when correct hand gesture is no longer detected.
        
        q = 5:
            (Turn in the XY plane)
            * Keeps the radius of the circle constant, but lets the operator
                turn the manipulator to the right/left according to the position
                of the operators hand in front of the camera.
            * TODO: Test moving up and down as well. Movement constrained to a
                cylinder instead of a circle
            * Will not allow operator to move end-effector in forbidden areas
    
    Update Operator panel:
    * Detected hand gesture
    * 
'''