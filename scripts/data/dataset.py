from pathlib import Path

import cv2
import numpy as np

from ..model import FaceDetector


class Video2Frame:
    """
    Converts video input data into frames and saves them.
    """

    def __init__(self, video_path: Path, save_dir: Path) -> None:
        """
        Args:
            video_path: Video file path (.mp4, .mov)
            save_dir: Parent folder path of the frames
        """
        self.video_path = video_path
        self.save_path = save_dir.joinpath(self.video_path.stem)

    def __call__(self) -> None:
        if not self.video_path.exists():
            raise ValueError("Video file does not exist.")

        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.video_path))

        frame_number: int = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_path = self.save_path / f"{self.video_path.stem}_{frame_number}.jpg"
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            frame_number += 1

        cap.release()


class FaceExtractor:
    """
    A class to detect face from frames using
    - (후보1) DeepFace(A hybrid face recognition framework wrapping SOTA detection model)
    - (후보2) YuNet(light-weight face detection model)
    - (후보3) YOLO face detection model

    original repo face detector model: S3FD, 사전 학습 weight 사용. 구현 사용 시, S3FD만 따로 학습 필요(torch 구현 실사용 어려움)
    """

    def __init__(self, detector_type: FaceDetector):
        self.model_type = detector_type

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray]:
        ret, faces = self.model_type.detector.detect(image)
        return (ret, faces)

    def extract(self, image: np.ndarray):
        ...
