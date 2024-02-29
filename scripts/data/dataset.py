from pathlib import Path

import cv2

class Video2Frame:
    """
    Converts video input data into frames and saves them.
    
    Args:
        video_path: Video file path (.mp4, .mov)
        save_dir: Parent folder path of the frames
    """
    def __init__(self, video_path: Path, save_dir: Path) -> None:
        self.video_path = video_path
        self.save_path = save_dir.joinpath(self.video_path.stem)

    def __call__(self):
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
        
