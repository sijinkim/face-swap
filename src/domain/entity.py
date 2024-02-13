from pydantic import BaseModel
from pathlib import Path

class Video(BaseModel):
    path: Path

class Frame(BaseModel):
    path: Path
    source_video_path: Path

class ExtractedFaceImage(BaseModel):
    path: Path
    source_frame_path: Path

class FaceMask(BaseModel):
    source_face: ExtractedFaceImage
