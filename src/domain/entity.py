from pydantic import BaseModel
from pathlib import Path

class Video(BaseModel):
    path: Path

class Frame(BaseModel):
    path: Path
    source_video_path: Video

class ExtractedFaceImage(BaseModel):
    path: Path
    source_frame_path: Frame

class FaceMask(BaseModel):
    source_face: ExtractedFaceImage
