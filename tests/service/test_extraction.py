from pathlib import Path

import pytest

from src.domain import Frame, Video
from src.service import get_frames_from_video, load_frames


@pytest.fixture
def video_path():
    return Path("./tests/test_data/test_video.mp4")


@pytest.fixture
def fail_video_path():
    return Path("./tests/test_data/fail_video.mp4")


def test_can_extract_frames_from_video(video_path):
    test_video = Video(path=video_path)
    get_frames_from_video(video=test_video)

    assert Path("./tests/test_data/test_video").exists()


def test_can_make_frame_object_using_frame_paths(video_path):
    test_frame_folder = video_path.parent / video_path.stem
    frames = load_frames(frame_folder=test_frame_folder)

    assert len(frames) == 15  # test_video.mp4 기준 추출 프레임 개수
    assert type(frames[0]) == Frame  # Frame entity 사용하여 로드
    assert frames[0].source_video_path == Video(path=video_path)  # source video 추적


def test_video_file_does_not_exist(fail_video_path):
    fail_video = Video(path=fail_video_path)

    with pytest.raises(ValueError) as excinfo:
        get_frames_from_video(video=fail_video)
    assert str(excinfo.value) == "Video file does not exist."


def test_frame_folder_does_not_exist(fail_video_path):
    fail_frame_folder = fail_video_path.parent / fail_video_path.stem

    with pytest.raises(ValueError) as excinfo:
        load_frames(frame_folder=fail_frame_folder)
    assert str(excinfo.value) == "Frame folder does not exist."
