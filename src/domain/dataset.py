from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
from skimage import transform
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


# Transforms - support Dataset bgr image preprocessing
class Rescale:
    def __init__(self, output_size: int) -> None:
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
        image = frame

        h, w = image.shape[:2]
        # 프레임의 짧은 면의 사이즈를 output_size에 맞춤
        if h > w:
            new_h, new_w = int(self.output_size * h / w), self.output_size
        else:
            new_h, new_w = self.output_size, int(self.output_size * h / w)

        resized_image: npt.NDArray[Any] = transform.resize(image, (new_h, new_w))  # type: ignore

        return resized_image


# convert ndarrays to tensors (H, W, C) -> (C, H, W)
class ToTensor:
    def __call__(self, frame: npt.NDArray[Any]) -> torch.Tensor:
        image = frame.transpose((2, 0, 1))

        return torch.from_numpy(image)
