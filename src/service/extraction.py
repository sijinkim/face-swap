from src.domain import Video, Frame
import cv2
from pathlib import Path

def get_frames_from_video(video: Video) -> None:
    """
        video를 입력받아 frame을 추출 후 저장함(jpg)
        추출 된 frame은 video가 위치한 곳에 video 이름과 동일한 폴더 하위에 저장됨

        Args:
            video: Video
        
        Returns:
    """
    video_path: Path = video.path

    if not video_path.exists():
        raise ValueError("Video file is not exist.")

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
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90 ] )

        frame_number += 1
    
    cap.release()
