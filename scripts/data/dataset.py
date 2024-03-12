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
    extract a face data from frames
    """

    face_detector: FaceDetector

    def detect_stage(self, image: np.ndarray) -> tuple[int, np.ndarray]:
        """
        detect face bbox and eyes, nose, mouth points.

        Args:
            image: an array of SINGLE image with A face (HWC BGR image - using cv2.imread)

        Returns:
            face: detection results - shape [num_faces, 15 face points info]
                  (here num_faces=1)
        """
        ret, face = self.face_detector.detector.detect(image)

        if not ret == 1:
            raise ValueError("Face detection failed.")

        return face

    def align_stage(
        self,
        image: np.ndarray,
        face: np.ndarray,
        desired_face_width: int = 256,
    ) -> np.ndarray:
        """
        Make the y-axis coordinate of the left and right eyes to the same level in the aligned image.

        Args:
            image: image to process
            face: face coordinates from the detect stage
            desired_face_width: The output width of the aligned face image

        Returns:
            np.ndarray: The processed image with aligned eye positions.
        """
        # apply same width, height
        h = w = desired_face_width

        # get coordinates of the center of the eyes
        # right_eye_x > left_eye_x
        right_eye = face[4:6]  # right_eye_x, right_eye_y
        left_eye = face[6:8]  # left_eye_x, left_eye_y
        eyes_center = (
            int((right_eye[0] + left_eye[0]) // 2),
            int((right_eye[1] + left_eye[1]) // 2),
        )

        # calculate the angle between center of the left eye and center of the right eye.
        # rotate the image based on this angle.
        delta_x = abs(int(left_eye[0] - right_eye[0]))
        delta_y = abs(int(left_eye[1] - right_eye[1]))
        angle = (np.arctan(delta_y / delta_x) * 180) / np.pi

        # define 2D matrix M
        M = cv2.getRotationMatrix2D(eyes_center, (angle), 1.0)
        # TODO: eye_center를 이미지 center로 맞추는 과정 필요. new_x, new_y

        # Applying the rotation to input image
        rotated = cv2.warpAffine(image, M, (new_x, new_y))
        return rotated
