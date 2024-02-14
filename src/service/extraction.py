from pathlib import Path

import cv2

from src.domain import Frame, Video


def get_frames_from_video(video: Video) -> None:
    video_path: Path = video.path

    if not video_path.exists():
        raise ValueError("Video file does not exist.")

    output_folder: Path = video_path.parent.joinpath(video_path.stem)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_number: int = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            # next frame 없는 경우
            break

        frame_path = output_folder / f"{video_path.stem}_{frame_number}.jpg"
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        frame_number += 1

    cap.release()


def load_frames(frame_folder: Path) -> list[Frame]:
    if not frame_folder.exists():
        raise ValueError("Frame folder does not exist.")

    frames: list[Frame] = []
    for frame in frame_folder.iterdir():
        frames.append(
            Frame(
                path=frame,
                source_video_path=Video(
                    path=frame_folder.parent / (frame_folder.name + ".mp4")
                ),
            )
        )

    return frames
