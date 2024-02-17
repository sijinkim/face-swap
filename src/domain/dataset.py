from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.domain import Frame, make_frame_list


class FrameDataset(Dataset):
    def __init__(self, data_folder: Path) -> None:
        self.data_paths: list[Frame] = make_frame_list(frame_folder=data_folder)

    def __getitem__(self, idx: int) -> np.ndarray:
        with open(self.data_paths[idx].path, "rb") as stream:
            bytes = bytearray(stream.read())
        image_np = np.asarray(bytes, dtype=np.uint8)

        return cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)

    def __len__(self) -> int:
        return len(self.data_paths)
