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
    Extract process
    1) rects_stage(): detect face position
    2) landmark_stage(): get face landmarks
    3) final_stage(): get aligned face data
    """

    model_type: FaceDetector

    def rects_stage(self, image: np.ndarray) -> tuple[int, np.ndarray[int,]]:
        """
        [Rects_stage]
        detect face bbox and eyes, nose, mouth points.

        Args:
            image: an array of SINGLE image with A face (HWC BGR image - using cv2.imread)

        Returns:
            face: detection results - shape [num_faces, 15 face points info]
                  (here num_faces=1)
        """
        ret, face = self.model_type.detector.detect(image)

        if not ret == 1:
            raise ValueError("Face detection failed.")

        return face

    def landmark_stage():
        ...

    def final_stage():
        ...
