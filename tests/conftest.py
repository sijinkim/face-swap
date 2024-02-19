from pathlib import Path

import pytest


@pytest.fixture
def video_path():
    return Path("./tests/test_data/test_video.mp4")


@pytest.fixture
def fail_video_path():
    return Path("./tests/test_data/fail_video.mp4")


@pytest.fixture
def frame_folder_path():
    return Path("./tests/test_data/test_video")
