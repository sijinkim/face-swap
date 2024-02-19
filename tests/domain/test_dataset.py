from src.domain.dataset import FrameDataset


def test_make_FrameDataset(frame_folder_path):
    sample_dataset = FrameDataset(data_folder=frame_folder_path)

    assert len(sample_dataset) == 15


def test_get_original_bgr_image_from_FrameDataset(frame_folder_path):
    sample_dataset = FrameDataset(data_folder=frame_folder_path)

    assert sample_dataset[0].shape == (1080, 1920, 3)
