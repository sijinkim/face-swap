from pathlib import Path

import cv2
import numpy as np
from model import FaceDetector, YuNet
from tqdm import tqdm


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

    def __call__(self) -> Path:
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

        print("Converting video to frames done.")
        return self.save_path


class FaceExtractor:
    """
    extract a face data from frames
    """

    def __init__(
        self,
        detector_type: str,
        input_dir: Path,
        output_image_width: int = 256,
    ) -> None:
        self.data_root_dir = input_dir
        self.image_paths = list(self.data_root_dir.glob("*.jpg"))

        sample_frame = cv2.imread(str(self.image_paths[0]))
        self.input_image_size = sample_frame.shape[:2]  # h,w

        if detector_type == "YuNet":
            self.face_detector: FaceDetector = YuNet(
                image_size=(self.input_image_size[1], self.input_image_size[0])
            )

        self.output_image_width = output_image_width

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

        return face

    def align_stage(
        self,
        image: np.ndarray,
        eyes_coord: np.ndarray,
        left_eye_desired_coordinate: np.ndarray = np.array((0.37, 0.37)),
    ) -> np.ndarray:
        """
        Make the y-axis coordinate of the left and right eyes to the same level in the aligned image.

        Args:
            image: image to process
            eyes_coord: a length 4 int garray of eyes coordinates from the detect stage
            left_eye_desired_coordinate: Fix coordinate ratio of left eye.

        Returns:
            np.ndarray: The processed image with aligned eye positions.
        """
        # apply same width, height
        desired_face_height = self.output_image_width

        # get desired right eye coordinate by using left eye
        # apply same y level (e.g. left eye: [0.37, 0.37], right eye: [0.63, 0.37])
        # final output image size를 1로 봤을 때, left eye/right eye 위치 (e.g. 3등분 -> 0.3x)
        right_eye_desired_coordinate = np.array(
            (1 - left_eye_desired_coordinate[0], left_eye_desired_coordinate[1])
        )

        # right_eye_x > left_eye_x
        right_eye = eyes_coord[:2]
        left_eye = eyes_coord[2:]

        # calculate the angle and distance between center of the left eye and center of the right eye.
        # rotate the image based on this angle.
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        dist_between_eyes = np.sqrt(delta_x**2 + delta_y**2)
        angles_between_eyes = (np.arctan(delta_y / delta_x) * 180) / np.pi
        eyes_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )

        # get scaled distance between eyes
        # final output image size(self.output_image_width/height) 반영한 left/right eyes 거리
        desired_dist_between_eyes = self.output_image_width * (
            right_eye_desired_coordinate[0] - left_eye_desired_coordinate[0]
        )
        scale = desired_dist_between_eyes / dist_between_eyes

        # define 2D matrix M
        M = cv2.getRotationMatrix2D(
            eyes_center, angles_between_eyes, scale
        )  # shape: (2, 3), affine matrix

        # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
        M[0, 2] += 0.5 * self.output_image_width - eyes_center[0]  # x transition
        M[1, 2] += (
            left_eye_desired_coordinate[1] * desired_face_height - eyes_center[1]
        )  # y transition

        # Applying the rotation to input image
        face_aligned = cv2.warpAffine(
            image,
            M,
            (self.output_image_width, desired_face_height),
            flags=cv2.INTER_CUBIC,
        )
        return face_aligned

    def extract_stage(self) -> None:
        """
        Extract detected aligned face and save to a directory(input_dir/aligned)

        Args:
            input_dir: path to the directory contains frames with single face.
            output_width: the width of final output image

        Returns:
            None
        """
        output_dir = self.data_root_dir / "aligned"
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(str(image_path))
            save_path = output_dir / image_path.name

            face_coordi = self.detect_stage(image=image)

            if face_coordi is None:
                print(f"Can't detect face at {image_path.name}.")
                continue

            face_aligned = self.align_stage(
                image=image, eyes_coord=list(map(lambda x: int(x), face_coordi[0][4:8]))
            )
            cv2.imwrite(
                str(save_path), face_aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
