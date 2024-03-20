import argparse
from pathlib import Path

import yaml

from data import FaceExtractor, Video2Frame


def preprocess(args):
    config_file = args.config_file
    assert (
        config_file.exists()
    ), f"The given config_file is not exists. {args.config_file}"

    with open(config_file) as config_io:
        config = yaml.load(config_io, Loader=yaml.BaseLoader)

    # step0. set data paths.
    src_video_path: Path = args.data_root / config["dataset"]["src_video_name"]
    dst_video_path: Path = args.data_root / config["dataset"]["dst_video_name"]
    save_root_path: Path = args.data_root

    # step1. Convert video to frames and get save path info.
    src_converter = Video2Frame(video_path=src_video_path, save_dir=save_root_path)
    dst_converter = Video2Frame(video_path=dst_video_path, save_dir=save_root_path)

    src_frame_path: Path = src_converter()
    dst_frame_path: Path = dst_converter()

    # step2. detect and extract aligned faces from frames.
    src_face_extractor = FaceExtractor(
        detector_type=config["models"]["detector"],
        input_dir=src_frame_path,
        output_image_width=config["dataset"]["image_width"],
    )
    dst_face_extractor = FaceExtractor(
        detector_type=config["models"]["detector"],
        input_dir=dst_frame_path,
        output_image_width=config["dataset"]["image_width"],
    )

    src_face_extractor.extract_stage()
    dst_face_extractor.extract_stage()


def train(args):
    ...


def inference(args):
    ...


def main():
    parser = argparse.ArgumentParser(prog="Face Swap implementation")
    sub_parser = parser.add_subparsers()

    preprocess_parser = sub_parser.add_parser("preprocess")
    preprocess_parser.set_defaults(func=preprocess)
    preprocess_parser.add_argument(
        "--config_file", type=Path, default="scripts_config.yaml"
    )
    preprocess_parser.add_argument("--data_root", type=Path, default="./data")

    train_parser = sub_parser.add_parser("train")
    train_parser.set_defaults(func=train)
    train_parser.add_argument("--config_file", type=Path, default="./config.yaml")

    inference_parser = sub_parser.add_parser("inference")
    inference_parser.set_defaults(func=inference)
    inference_parser.add_argument("--config_file", type=Path, default="./config.yaml")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
