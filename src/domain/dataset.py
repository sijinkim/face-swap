from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from .entity import Frame
from .extraction import make_frame_list


class FrameDataset(Dataset[npt.NDArray[Any]]):
    data_paths: list[Frame]

    def __init__(self, data_folder: Path) -> None:
        assert isinstance(data_folder, Path)
        self.data_paths: list[Frame] = make_frame_list(frame_folder=data_folder)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> npt.NDArray[Any]:
        with open(self.data_paths[idx].path, "rb") as stream:
            _bytes = bytearray(stream.read())  # read image file by bytes
        image_np = np.asarray(_bytes, dtype=np.uint8)  # take 8bits

        bgr_image = cv2.imdecode(buf=image_np, flags=cv2.IMREAD_UNCHANGED)  # matrix
        return bgr_image
