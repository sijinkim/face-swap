import numpy as np
import pytest
import torch

from src.domain.dataset import FrameDataset, Rescale, ToTensor


@pytest.fixture
def sample_dataset(frame_folder_path):
    return FrameDataset(data_folder=frame_folder_path)


def test_make_FrameDataset(sample_dataset):
    sample_dataset = sample_dataset

    assert len(sample_dataset) == 15


def test_get_original_bgr_image_from_FrameDataset(sample_dataset):
    sample_dataset = sample_dataset

    assert sample_dataset[0].shape == (1080, 1920, 3)


def test_rescale_frame_data(sample_dataset):
    sample_dataset = sample_dataset

    h, w = sample_dataset[0].shape[:2]
    rescale = Rescale(output_size=512)
    assert rescale(sample_dataset[0]).shape == (512, int(512 * h / w), 3)


def test_convert_ndarray_to_tensor(sample_dataset):
    sample_dataset = sample_dataset
    totensor = ToTensor()

    image_array = sample_dataset[0]
    assert isinstance(image_array, np.ndarray)
    assert torch.is_tensor(totensor(image_array))
