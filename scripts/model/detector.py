from pathlib import Path

import cv2
import numpy as np
import torch.nn as nn


class FaceDetector:
    ...


class S3FD(FaceDetector):
    ...


class YOLO5Face(FaceDetector):
    ...


class YuNet(FaceDetector):
    def __init__(self, image_size: tuple[int, int], model_path: Path):
        """
        Args:
            image_size: a tuple of height, width of the image
            model_path: a path of pretrained YuNet model. (~/model/weights/face_detection_yunet_2023mar.onnx)
        """
        face_detection_model = model_path
        if not face_detection_model.exists():
            raise ValueError("YuNet model weights does not exist.")

        self.detector = cv2.FaceDetectorYN.create(
            model=str(face_detection_model), config="", input_size=image_size
        )