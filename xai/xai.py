from sys import path
import numpy as np
import mediapipe as mp
import tensorflow as tf
import cv2

class XAI:
    def __init__(self, pathToModel, pathToDataset):
        self.pathToModel = pathToModel
        self.pathToDataset = pathToDataset

    def loadModel(self):
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=self.pathToModel)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def process(self):
        # Test model on random input data.
        img = cv2.imread(f"{self.pathToDataset}1.jpg")
        input_shape = self.input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # mp.calculators.tensor.image_to_tensor_calculator_pb2.ImageToTensorCalculatorOptions

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(output_data)


if __name__ == "__main__":
    xai = XAI("data/palm_detection_full.tflite", "data/dataset1/")
    xai.loadModel()
    xai.process()