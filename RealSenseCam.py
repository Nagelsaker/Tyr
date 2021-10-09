import pyrealsense2 as rs
import numpy as np
import cv2


class CameraStream:
    def __init__(self):
        self.pipeline = rs.pipeline()

        # Configure streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    def start(self):
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depthSensor = self.profile.get_device().first_depth_sensor()
        self.depthScale = depthSensor.get_depth_scale()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        alignTo = rs.stream.color
        self.align = rs.align(alignTo)

    def end(self):
        self.pipeline.stop()

    def getImages(self):
        '''
        Returns raw color and depth images

        Out:
            images: (H x 2*W x 3 Array)
                Color image and depth image horizontally stacked
        '''
        frames = self.pipeline.wait_for_frames()
        depthFrame = frames.get_depth_frame()
        colorFrame = frames.get_color_frame()

        if not depthFrame or not colorFrame:
            return

        # Convert images to numpy arrays
        depthImage = np.asanyarray(depthFrame.get_data())
        colorImage = np.asanyarray(colorFrame.get_data())

        # depthColormap = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((colorImage, depthImage))
        return images

    def getAlignedImages(self, clippingDistanceInMeters=0):
        '''
        Aligns the depth image with the RGB image. Useful for extracting depth from detections.

        In:
            clippingDistanceInMeters: Int
                Colors all pixels outside of the clipping distance grey
        
        Out:
            images: (H x 2*W x 3) Array
                Color image and depth image horizontally stacked
        '''
        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clippingDistance = clippingDistanceInMeters / self.depthScale
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        alignedFrames = self.align.process(frames)

        # Get aligned frames
        alignedDepthFrame = alignedFrames.get_depth_frame()
        colorFrame = alignedFrames.get_color_frame()

        # Validate that both frames are valid
        if not alignedDepthFrame or not colorFrame:
            return -1

        depthImage = np.asanyarray(alignedDepthFrame.get_data())
        colorImage = np.asanyarray(colorFrame.get_data())

        depthImage3d = np.dstack((depthImage,depthImage,depthImage)) #depth image is 1 channel, color is 3 channels
        # Remove background - Set pixels further than clipping_distance to grey
        if type(clippingDistance) is np.ndarray:
            grey_color = 153
            colorImage = np.where((depthImage3d > clippingDistance[1]) | (depthImage3d <= clippingDistance[0]), grey_color, colorImage)

        depthColormap = self.getColorMap(depthImage)

        depthImgInMeter = (depthImage3d * self.depthScale)

        images = np.hstack((colorImage, depthImgInMeter))
        return images
    
    def getColorMap(self, depthImage):
        depthColormap = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.03), cv2.COLORMAP_JET)
        return depthColormap

    def showImages(self, images):
        '''
        Render images:
            depth align to color on left
            depth on right

        In:
            images: (H x 2*W x 3) Array
                Color image and depth image horizontally stacked
        '''
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return -1
        else:
            return 0


if __name__ == "__main__":
    cs = CameraStream()
    cs.start()
    try:
        while True:
            cs.getAlignedImages()
    finally:
        cs.end()
    # while True:
    #     cs.getFrame()
    # cs.end()